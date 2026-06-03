#!/usr/bin/env python3
"""
Akiti-TTS — local CPU inference.

Generates Asante Twi speech on your own CPU using the GGUF model. The model is
downloaded from HuggingFace on first run and cached (~/.cache/huggingface),
so subsequent runs are offline-capable. Voice presets ship in this repo
(voices.json) so cloning gives you everything except the (large) model weights.

Why a custom tokenizer step? The GGUF's embedded tokenizer is lossy, so we
tokenize with the ORIGINAL training tokenizer and feed token IDs straight to
llama.cpp; only the GGUF *weights* are used.

------------------------------------------------------------------------------
Install (one time)
------------------------------------------------------------------------------
System dependency — espeak-ng:
    Linux:   sudo apt-get install espeak-ng
    macOS:   brew install espeak-ng
    Windows: download from https://github.com/espeak-ng/espeak-ng/releases

Python packages:
    pip install -r requirements.txt

------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------
    python akiti_local.py --list-voices
    python akiti_local.py --text "Meda wo ase paa." --output out.wav
    python akiti_local.py --text "Akwaaba!" --voice kofi --output out.wav
    python akiti_local.py --text "Meda wo ase." --model q8 --output out.wav

Anonymous usage stats (opt-in): on first run you'll be asked whether to share
anonymous performance metrics (CPU type, generation speed — never your text or
audio). Use --no-stats to skip for a run, or --reset-stats to be asked again.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import re
import sys
import time

MODEL_REPO     = "michsethowusu/Akiti-TTS"
CODEC_REPO     = "neuphonic/neucodec-onnx-decoder-int8"
GGUF_FILES     = {"q4": "VieNeu-TTS-Twi-Q4_K_M.gguf", "q8": "VieNeu-TTS-Twi-Q8_0.gguf"}
SAMPLE_RATE    = 24000
MAX_REF_CODES  = 200
SCRIPT_VERSION = "1.0.0"
STATS_URL      = "https://ghana-nlp--akiti-tts-submit-stats.modal.run"
CONFIG_PATH    = os.path.join(os.path.expanduser("~"), ".config", "akiti-tts", "config.json")
HERE           = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Telemetry helpers (opt-in, anonymous)
# ---------------------------------------------------------------------------
def _cpu_name() -> str:
    s = platform.processor()
    try:
        if not s and platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
        if not s and platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return s or platform.machine() or "unknown"


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def telemetry_enabled(reset: bool = False) -> bool:
    """Return whether to send stats, prompting once on first run."""
    cfg = _load_config()
    if reset:
        cfg.pop("telemetry", None)
    if "telemetry" in cfg:
        return bool(cfg["telemetry"])

    # First run — ask, but only if interactive; otherwise default off.
    if not sys.stdin.isatty():
        cfg["telemetry"] = False
        _save_config(cfg)
        return False

    print("\n" + "-" * 70)
    print("Help improve Akiti-TTS?")
    print("Share ANONYMOUS performance stats — your CPU model and generation")
    print("speed. Your text and audio are NEVER sent. This helps build a table")
    print("of how fast the model runs on different CPUs.")
    ans = input("Share anonymous stats? [y/N]: ").strip().lower()
    print("-" * 70 + "\n")
    cfg["telemetry"] = ans in ("y", "yes")
    _save_config(cfg)
    return cfg["telemetry"]


def send_stats(stats: dict) -> None:
    """Best-effort POST; never raises, runs in a background thread."""
    import threading

    def _post():
        try:
            import urllib.request
            data = json.dumps(stats).encode("utf-8")
            req = urllib.request.Request(STATS_URL, data=data,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass  # telemetry must never affect the user

    threading.Thread(target=_post, daemon=True).start()


# ---------------------------------------------------------------------------
# TTS engine
# ---------------------------------------------------------------------------
class AkitiTTS:
    def __init__(self, model: str = "q4", n_threads: int | None = None):
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from llama_cpp import Llama
        from phonemizer import phonemize as _ph

        t0 = time.time()
        self.np, self.re, self._ph = np, re, _ph

        print("Loading tokenizer ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
        self.speech_end_id = self.tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        print("Loading ONNX codec ...", flush=True)
        onnx_path = hf_hub_download(repo_id=CODEC_REPO, filename="model.onnx")
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.codec = onnxruntime.InferenceSession(onnx_path, sess_options=so,
                                                  providers=["CPUExecutionProvider"])

        # Prefer the voices.json bundled in this repo; fall back to HF.
        local_voices = os.path.join(HERE, "voices.json")
        vpath = local_voices if os.path.exists(local_voices) else \
            hf_hub_download(repo_id=MODEL_REPO, filename="voices.json")
        self.voices = json.load(open(vpath, encoding="utf-8"))
        self._ref_cache: dict[str, str] = {}

        gguf_file = GGUF_FILES.get(model, GGUF_FILES["q4"])
        print(f"Loading GGUF weights ({gguf_file}) — first run downloads the model ...", flush=True)
        gguf_path = hf_hub_download(repo_id=MODEL_REPO, filename=gguf_file)
        self.threads = n_threads or max(multiprocessing.cpu_count(), 1)
        self.llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=0,
                         n_threads=self.threads, n_threads_batch=self.threads, verbose=False)
        self.load_seconds = round(time.time() - t0, 2)
        print(f"Ready in {self.load_seconds}s. (CPU threads: {self.threads})", flush=True)

    def voice_names(self) -> list[str]:
        return list(self.voices["presets"].keys())

    def _phon(self, text: str) -> str:
        return self._ph(text, backend="espeak", language="lfn",
                        with_stress=True, preserve_punctuation=True)

    def _gen_codes(self, prompt, temperature, top_p, rep_pen, max_tokens=600) -> list[int]:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        out: list[int] = []
        for tid in self.llm.generate(ids, temp=temperature, top_p=top_p, top_k=50,
                                     repeat_penalty=rep_pen):
            if tid == self.speech_end_id or tid == self.tok.eos_token_id:
                break
            out.append(tid)
            if len(out) >= max_tokens:
                break
        toks = self.tok.convert_ids_to_tokens(out)
        return [int(m.group(1)) for t in toks
                for m in [self.re.match(r"<\|speech_(\d+)\|>", t or "")] if m]

    def _decode(self, codes):
        arr = self.np.array(codes, dtype=self.np.int32)[None, None, :]
        recon = self.codec.run(None, {"codes": arr})[0]
        return self.np.asarray(recon).reshape(-1).astype("float32")

    def infer(self, text, voice="none", temperature=0.4, top_p=0.8, rep_pen=1.3):
        """Returns (wav, n_codes)."""
        ref_codes = ref_phones = None
        if voice and voice != "none":
            if voice not in self.voices["presets"]:
                raise ValueError(f"Voice '{voice}' not found. Available: {self.voice_names()}")
            v = self.voices["presets"][voice]
            ref_codes = v["codes"][:MAX_REF_CODES]
            if voice not in self._ref_cache:
                self._ref_cache[voice] = self._phon(v["text"])
            ref_phones = self._ref_cache[voice]

        sentences = [s.strip() for s in self.re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()] or [text]
        all_codes: list[int] = []
        for sent in sentences:
            tgt = self._phon(sent).strip()
            if ref_codes is not None:
                codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
                prompt = (f"<|TEXT_PROMPT_START|>{ref_phones.strip()} {tgt}"
                          f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}")
            else:
                prompt = f"<|TEXT_PROMPT_START|>{tgt}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>"
            all_codes.extend(self._gen_codes(prompt, temperature, top_p, rep_pen))

        if not all_codes:
            raise RuntimeError("No speech generated — try lowering --temperature or another voice.")
        return self._decode(all_codes), len(all_codes)


def main():
    p = argparse.ArgumentParser(description="Akiti-TTS local CPU inference (Asante Twi).")
    p.add_argument("--text", help="Twi text to synthesize")
    p.add_argument("--voice", default="none", help="Voice preset name, or 'none' (default speaker)")
    p.add_argument("--model", default="q4", choices=["q4", "q8"], help="q4 = faster, q8 = better quality")
    p.add_argument("--output", default="output.wav", help="Output WAV path")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--rep-penalty", type=float, default=1.3, help="Higher reduces silences/repeats")
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores)")
    p.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    p.add_argument("--no-stats", action="store_true", help="Don't send anonymous stats this run")
    p.add_argument("--reset-stats", action="store_true", help="Re-ask the stats consent question")
    args = p.parse_args()

    if args.list_voices:
        tts = AkitiTTS(model=args.model, n_threads=args.threads)
        print("\nAvailable voices:")
        for v in tts.voice_names():
            print(f"  - {v}")
        print("  - none  (default speaker, no reference voice)")
        return

    if not args.text:
        p.error("--text is required (or use --list-voices)")

    import soundfile as sf
    tts = AkitiTTS(model=args.model, n_threads=args.threads)
    print(f"Synthesizing: {args.text!r}  (voice={args.voice})", flush=True)

    t0 = time.time()
    wav, n_codes = tts.infer(args.text, voice=args.voice, temperature=args.temperature,
                             top_p=args.top_p, rep_pen=args.rep_penalty)
    gen_seconds = round(time.time() - t0, 2)
    audio_seconds = round(len(wav) / SAMPLE_RATE, 2)

    sf.write(args.output, wav, SAMPLE_RATE)
    rtf = round(gen_seconds / audio_seconds, 2) if audio_seconds else None
    print(f"Saved {audio_seconds}s -> {os.path.abspath(args.output)}  "
          f"(gen {gen_seconds}s, RTF {rtf}x)")

    # Opt-in anonymous telemetry
    if not args.no_stats and telemetry_enabled(reset=args.reset_stats):
        send_stats({
            "cpu": _cpu_name(),
            "arch": platform.machine(),
            "os": platform.system(),
            "python": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(),
            "threads": tts.threads,
            "model": args.model,
            "speech_codes": n_codes,
            "audio_seconds": audio_seconds,
            "gen_seconds": gen_seconds,
            "load_seconds": tts.load_seconds,
            "rtf": rtf,
            "script_version": SCRIPT_VERSION,
        })


if __name__ == "__main__":
    main()
