#!/usr/bin/env python3
"""
Akiti-TTS REST API — local HTTP server for programmatic inference.

Exposes the same warm AkitiTTS engine used by akiti_local.py / akiti_web.py
over a small JSON/REST API, so you can synthesize Asante Twi speech from any
language or tool (curl, Python requests, n8n, etc.). The model loads once on
startup and stays warm; each request is just the synthesis time.

Long text is split on sentence boundaries into chunks (<= MAX_CHUNK_CHARS),
synthesized one chunk at a time, then stitched — same as the web Studio, but
returned synchronously in a single response.

Everything runs locally (aside from the opt-in anonymous performance stats
described in the README; opt out with AKITI_NO_STATS=1).

Run:
    python akiti_api.py                      # loads q8, serves on 127.0.0.1:8800
    python akiti_api.py --model q4           # faster, lower quality
    python akiti_api.py --host 0.0.0.0 --port 9000 --cors
    AKITI_API_KEY=secret python akiti_api.py # require an API key

Endpoints:
    GET  /health                 -> engine status
    GET  /v1/voices              -> available voice presets
    POST /v1/synthesize          -> WAV audio (or JSON with base64 audio)

Synthesize (binary WAV back):
    curl -s -X POST http://127.0.0.1:8800/v1/synthesize \
         -H 'Content-Type: application/json' \
         -d '{"text": "Meda wo ase paa."}' --output out.wav

Synthesize (JSON back, base64 audio + metadata):
    curl -s -X POST 'http://127.0.0.1:8800/v1/synthesize?format=json' \
         -H 'Content-Type: application/json' \
         -d '{"text": "Akwaaba!", "model": "q8"}'

Request body fields (all optional except text):
    text         string  (required)
    model        "q4" | "q8"        default: server's --model
    voice        preset name | "none"   default: "none"
    temperature  float   default: 0.4
    rep_penalty  float   default: 1.3
    top_p        float   default: 0.8
"""
from __future__ import annotations

import argparse
import base64
import io
import multiprocessing
import os
import platform
import threading
import time

from flask import Flask, jsonify, request, send_file

from akiti_local import (AkitiTTS, SAMPLE_RATE, SCRIPT_VERSION, MAX_CHUNK_CHARS,
                         chunk_text, stitch_wavs, send_stats, _cpu_name)

app = Flask(__name__)

# --- Engine state (single warm engine, serialized access) ------------------
_lock = threading.Lock()
_engine: AkitiTTS | None = None
_engine_model: str | None = None
STATE = {"ready": False, "loading": True, "model": None,
         "error": None, "load_seconds": None, "threads": None}


def _ensure_engine_locked(model: str) -> AkitiTTS:
    """(Re)load the engine for `model`. Caller MUST hold _lock."""
    global _engine, _engine_model
    if _engine is not None and _engine_model == model:
        return _engine
    STATE.update(ready=False, loading=True, model=model, error=None)
    try:
        eng = AkitiTTS(model=model)
    except Exception as e:  # noqa: BLE001
        STATE.update(loading=False, error=str(e))
        raise
    _engine, _engine_model = eng, model
    STATE.update(ready=True, loading=False, model=model,
                 load_seconds=eng.load_seconds, threads=eng.threads)
    return eng


def _warm_up(model: str):
    """Background pre-load so /health responds while the model loads."""
    try:
        with _lock:
            _ensure_engine_locked(model)
    except Exception:
        pass  # error is recorded in STATE and surfaced via /health


# --- Auth / CORS -----------------------------------------------------------
def _check_auth() -> bool:
    """True if no API key is configured, or the request presents the right one."""
    key = app.config.get("API_KEY")
    if not key:
        return True
    header = request.headers.get("Authorization", "")
    bearer = header[7:] if header.startswith("Bearer ") else None
    return key in (bearer, request.headers.get("X-API-Key"))


@app.after_request
def _add_cors(resp):
    if app.config.get("CORS"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


# --- Routes ----------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify(version=SCRIPT_VERSION + "-api", **STATE)


@app.route("/v1/voices")
def voices():
    if not _check_auth():
        return jsonify(error="Unauthorized."), 401
    with _lock:
        try:
            eng = _ensure_engine_locked(app.config["DEFAULT_MODEL"])
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e)), 503
        names = eng.voice_names()
    return jsonify(voices=["none"] + names, default="none")


@app.route("/v1/synthesize", methods=["POST", "OPTIONS"])
def synthesize():
    if request.method == "OPTIONS":
        return ("", 204)
    if not _check_auth():
        return jsonify(error="Unauthorized."), 401

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Field 'text' is required."), 400

    model = data.get("model") or app.config["DEFAULT_MODEL"]
    if model not in ("q4", "q8"):
        return jsonify(error="Field 'model' must be 'q4' or 'q8'."), 400
    voice = data.get("voice") or "none"
    try:
        temperature = float(data.get("temperature", 0.4))
        top_p = float(data.get("top_p", 0.8))
        rep_pen = float(data.get("rep_penalty", 1.3))
    except (TypeError, ValueError):
        return jsonify(error="temperature/top_p/rep_penalty must be numbers."), 400

    chunks = chunk_text(text)
    if not chunks:
        return jsonify(error="Field 'text' is required."), 400

    t0 = time.time()
    try:
        with _lock:
            eng = _ensure_engine_locked(model)
            wavs, total_codes = [], 0
            for chunk in chunks:
                wav, n_codes = eng.infer(chunk, voice=voice, temperature=temperature,
                                         top_p=top_p, rep_pen=rep_pen)
                wavs.append(wav)
                total_codes += n_codes
            full = stitch_wavs(wavs)
    except ValueError as e:  # bad voice name, etc. -> client error
        return jsonify(error=str(e)), 400
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 500

    gen = round(time.time() - t0, 2)
    audio = round(len(full) / SAMPLE_RATE, 2)
    rtf = round(gen / audio, 2) if audio else None

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, full, SAMPLE_RATE, format="WAV")
    wav_bytes = buf.getvalue()

    # Anonymous performance telemetry (opt out with AKITI_NO_STATS=1).
    send_stats({
        "cpu": _cpu_name(), "arch": platform.machine(), "os": platform.system(),
        "python": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "threads": eng.threads, "model": model,
        "speech_codes": total_codes, "audio_seconds": audio, "gen_seconds": gen,
        "load_seconds": eng.load_seconds, "rtf": rtf, "chunks": len(chunks),
        "script_version": SCRIPT_VERSION + "-api",
    })

    # JSON response (base64 audio + metadata) when explicitly requested,
    # otherwise stream the raw WAV with the same metadata in X- headers.
    wants_json = (request.args.get("format") == "json"
                  or request.args.get("encoding") == "base64")
    if wants_json:
        return jsonify(
            audio_base64=base64.b64encode(wav_bytes).decode("ascii"),
            format="wav", sample_rate=SAMPLE_RATE, model=model,
            chunks=len(chunks), gen_seconds=gen, audio_seconds=audio, rtf=rtf,
        )

    resp = send_file(io.BytesIO(wav_bytes), mimetype="audio/wav",
                     download_name="akiti.wav")
    resp.headers["X-Gen-Seconds"] = str(gen)
    resp.headers["X-Audio-Seconds"] = str(audio)
    resp.headers["X-RTF"] = str(rtf)
    resp.headers["X-Model"] = model
    resp.headers["X-Chunks"] = str(len(chunks))
    return resp


def main():
    p = argparse.ArgumentParser(description="Akiti-TTS REST API server.")
    p.add_argument("--model", default="q8", choices=["q4", "q8"],
                   help="q8 = better quality, q4 = faster (default: q8)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind host (use 0.0.0.0 to expose on your network)")
    p.add_argument("--port", type=int, default=8800, help="Bind port")
    p.add_argument("--api-key", default=os.environ.get("AKITI_API_KEY"),
                   help="Require this key via 'Authorization: Bearer <key>' "
                        "or 'X-API-Key' (or set AKITI_API_KEY)")
    p.add_argument("--cors", action="store_true",
                   help="Send permissive CORS headers (allow any origin)")
    args = p.parse_args()

    app.config["DEFAULT_MODEL"] = args.model
    app.config["API_KEY"] = args.api_key
    app.config["CORS"] = args.cors

    threading.Thread(target=_warm_up, args=(args.model,), daemon=True).start()

    print(f"\nAkiti-TTS API → http://{args.host}:{args.port}  "
          f"(loading {args.model} model…)")
    if args.api_key:
        print("API key required (Authorization: Bearer <key> or X-API-Key).")
    print(f"  GET  /health\n  GET  /v1/voices\n  POST /v1/synthesize\n")

    # Threaded so /health stays responsive while a generation runs.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
