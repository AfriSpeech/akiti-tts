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

Anonymous usage stats: each run sends anonymous performance metrics (CPU model,
generation speed — never your text or audio) to help build a public benchmark of
how fast the model runs on different CPUs. Opt out with --no-stats, or by setting
the environment variable AKITI_NO_STATS=1.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import platform
import re
import time

MODEL_REPO     = "afrispeech/Akiti-TTS"
CODEC_REPO     = "neuphonic/neucodec-onnx-decoder-int8"
GGUF_FILES     = {"q4": "VieNeu-TTS-Twi-Q4_K_M.gguf", "q8": "VieNeu-TTS-Twi-Q8_0.gguf"}
SAMPLE_RATE    = 24000
MAX_REF_CODES  = 200
SCRIPT_VERSION = "1.0.0"
STATS_URL      = "https://ghana-nlp--akiti-tts-submit-stats.modal.run"
HERE           = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def _suppress_stderr():
    """Silence C-level stderr (e.g. llama.cpp's n_ctx notice) at the fd level.

    verbose=False doesn't catch messages emitted directly to fd 2 by the
    underlying C library, so we redirect the fd to /dev/null for the block.
    """
    saved_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)


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


def send_stats(stats: dict) -> None:
    """Best-effort POST of anonymous metrics; never raises, runs in a thread.

    Auto-sent by default. Set AKITI_NO_STATS=1 (or pass --no-stats on the CLI)
    to opt out — see the README's "Anonymous usage stats" section for what is
    and isn't sent, and why it helps.
    """
    if os.environ.get("AKITI_NO_STATS"):
        return
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
# Chunked synthesis helpers (shared by the web + REST API servers)
# ---------------------------------------------------------------------------
# Long text is split into chunks no larger than MAX_CHUNK_CHARS, on sentence
# boundaries, and synthesized one chunk at a time. This keeps each generation
# bounded and lets the servers report progress. GAP_SECONDS of silence is
# inserted between stitched chunks so the joins don't sound abrupt.
MAX_CHUNK_CHARS = 200
GAP_SECONDS = 0.2
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break a single over-long sentence into word-aligned pieces."""
    pieces: list[str] = []
    cur = ""
    for word in sentence.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            pieces.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Group whole sentences into chunks no longer than `max_chars`.

    Short text returns a single chunk. A sentence longer than the limit is
    split on word boundaries so no chunk ever exceeds it.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()] or [text]
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_long_sentence(sent, max_chars))
            continue
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def stitch_wavs(wavs: list, gap_seconds: float = GAP_SECONDS):
    """Concatenate per-chunk waveforms with a short silence between each."""
    import numpy as np
    if not wavs:
        raise ValueError("No audio to stitch.")
    if len(wavs) == 1:
        return wavs[0]
    gap = np.zeros(int(SAMPLE_RATE * gap_seconds), dtype=wavs[0].dtype)
    joined: list = []
    for i, w in enumerate(wavs):
        if i:
            joined.append(gap)
        joined.append(w)
    return np.concatenate(joined)


# ---------------------------------------------------------------------------
# TTS engine
# ---------------------------------------------------------------------------
class AkitiTTS:
    def __init__(self, model: str = "q4", n_threads: int | None = None):
        import numpy as np
        # onnxruntime probes for GPUs at import and warns on CPU-only boxes;
        # silence that fd-level chatter during the import itself.
        with _suppress_stderr():
            import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from llama_cpp import Llama
        from phonemizer import phonemize as _ph

        onnxruntime.set_default_logger_severity(3)  # 3 = error (hide warnings)

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
        with _suppress_stderr():
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
    p.add_argument("--no-stats", action="store_true",
                   help="Opt out of anonymous performance stats for this run")
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

    # Anonymous performance telemetry — auto-sent; opt out with --no-stats
    # or AKITI_NO_STATS=1 (handled inside send_stats).
    if not args.no_stats:
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
