#!/usr/bin/env python3
"""
Akiti-TTS Studio — local web app.

Spins up a small Flask server and opens a browser UI for creating voice content.
Akiti TTS currently supports Asante Twi, with more languages planned. It wraps
the same AkitiTTS engine used by akiti_local.py: the model loads once on startup
and stays warm, so each generation is just the synthesis time.

Everything runs locally on your machine — no audio or text leaves your computer
(aside from the opt-in anonymous performance stats described in the README).

Run:
    python akiti_web.py                  # loads q8, opens http://127.0.0.1:7860
    python akiti_web.py --model q4        # faster, lower quality
    python akiti_web.py --port 8000 --no-browser

Opt out of anonymous stats with the AKITI_NO_STATS=1 environment variable.
"""
from __future__ import annotations

import argparse
import io
import multiprocessing
import platform
import threading
import time
import webbrowser

from flask import Flask, jsonify, render_template, request, send_file

from akiti_local import (AkitiTTS, SAMPLE_RATE, SCRIPT_VERSION,
                         send_stats, _cpu_name)

# Voice cloning isn't production-ready yet, so the voice picker is hidden for
# now and every generation uses the default speaker. The engine still supports
# voices (see AkitiTTS.infer) — flip this to True to surface the picker.
VOICE_SELECTION_ENABLED = False

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
    """Background pre-load so the page can render while the model loads."""
    try:
        with _lock:
            _ensure_engine_locked(model)
    except Exception:
        pass  # error is recorded in STATE and surfaced via /api/status


# --- Routes ----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           voice_enabled=VOICE_SELECTION_ENABLED,
                           default_model=app.config["DEFAULT_MODEL"])


@app.route("/api/status")
def status():
    return jsonify(STATE)


@app.route("/api/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Please enter some text."), 400

    model = data.get("model") or app.config["DEFAULT_MODEL"]
    if model not in ("q4", "q8"):
        model = app.config["DEFAULT_MODEL"]
    try:
        temperature = float(data.get("temperature", 0.4))
        rep_pen = float(data.get("rep_penalty", 1.3))
    except (TypeError, ValueError):
        return jsonify(error="Invalid temperature or rep_penalty."), 400

    voice = "none"
    if VOICE_SELECTION_ENABLED:
        voice = data.get("voice") or "none"

    t0 = time.time()
    try:
        with _lock:
            eng = _ensure_engine_locked(model)
            wav, n_codes = eng.infer(text, voice=voice,
                                     temperature=temperature, rep_pen=rep_pen)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 500

    gen = round(time.time() - t0, 2)
    audio = round(len(wav) / SAMPLE_RATE, 2)
    rtf = round(gen / audio, 2) if audio else None

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV")
    buf.seek(0)

    # Anonymous performance telemetry (opt out with AKITI_NO_STATS=1).
    send_stats({
        "cpu": _cpu_name(), "arch": platform.machine(), "os": platform.system(),
        "python": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "threads": eng.threads, "model": model,
        "speech_codes": n_codes, "audio_seconds": audio, "gen_seconds": gen,
        "load_seconds": eng.load_seconds, "rtf": rtf,
        "script_version": SCRIPT_VERSION + "-web",
    })

    resp = send_file(buf, mimetype="audio/wav")
    resp.headers["X-Gen-Seconds"] = str(gen)
    resp.headers["X-Audio-Seconds"] = str(audio)
    resp.headers["X-RTF"] = str(rtf)
    resp.headers["X-Model"] = model
    return resp


def main():
    p = argparse.ArgumentParser(description="Akiti-TTS Studio web app.")
    p.add_argument("--model", default="q8", choices=["q4", "q8"],
                   help="q8 = better quality, q4 = faster (default: q8)")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=7860, help="Bind port")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the browser")
    args = p.parse_args()

    app.config["DEFAULT_MODEL"] = args.model
    threading.Thread(target=_warm_up, args=(args.model,), daemon=True).start()

    url = f"http://{args.host}:{args.port}"
    print(f"\nAkiti-TTS Studio → {url}  (loading {args.model} model…)\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    # Threaded so /api/status stays responsive while a generation runs.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
