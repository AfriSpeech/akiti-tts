# 🪘 Akiti TTS

Asante Twi text-to-speech you can run **locally on your own CPU**. Akiti TTS is
fine-tuned from [VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
and served as a small quantized GGUF model, so it runs without a GPU.

- 🧠 Model & voices: [michsethowusu/Akiti-TTS](https://huggingface.co/michsethowusu/Akiti-TTS)
- 🌐 Web demo: [michsethowusu/Akiti-TTS Space](https://huggingface.co/spaces/michsethowusu/Akiti-TTS)

The model weights download automatically from HuggingFace on first run and are
cached locally afterward. The voice presets (`voices.json`) ship with this repo.

## Install

**1. System dependency — espeak-ng**

```bash
# Linux
sudo apt-get install espeak-ng
# macOS
brew install espeak-ng
# Windows: https://github.com/espeak-ng/espeak-ng/releases
```

**2. Python packages**

```bash
pip install -r requirements.txt
```

> Note: `llama-cpp-python` may build from source on first install (a few
> minutes). Prebuilt CPU wheels are available for some platforms.

## Usage

```bash
# List the available voices
python akiti_local.py --list-voices

# Default speaker (no reference voice)
python akiti_local.py --text "Meda wo ase paa." --output out.wav

# Use a voice preset
python akiti_local.py --text "Akwaaba! Wo ho te sɛn?" --voice kofi --output out.wav

# Higher quality (Q8) model
python akiti_local.py --text "Meda wo ase." --model q8 --output out.wav
```

### Studio (web app)

For creating voice content, run the browser-based **Akiti TTS Studio**:

```bash
python akiti_web.py
```

It starts a small local server and opens `http://127.0.0.1:7860` in your
browser. The model loads once and stays warm, so each generation is fast. From
there you can:

- write or paste a script and **Generate** (or press **Ctrl/⌘ + Enter**),
- play clips inline and **Download** them as WAV,
- build up a session **library** of generated clips,
- tune the model (q4/q8), temperature, and repetition penalty.

Options: `--model q8`, `--port 8000`, `--host 0.0.0.0`, `--no-browser`.
Everything runs locally — nothing is uploaded.

> Voice selection is **temporarily disabled** while voice cloning is being
> improved, so generation uses the default speaker. (The CLI still exposes
> `--voice` / `--list-voices` for testing.)

### Options

| Flag | Default | Description |
|---|---|---|
| `--text` | — | Twi text to synthesize |
| `--voice` | `none` | Voice preset name, or `none` for the default speaker |
| `--model` | `q4` | `q4` (faster) or `q8` (better quality) |
| `--output` | `output.wav` | Output WAV path |
| `--temperature` | `0.4` | Higher = more varied |
| `--top-p` | `0.8` | Nucleus sampling |
| `--rep-penalty` | `1.3` | Higher reduces silences/repeats |
| `--threads` | all cores | CPU threads to use |
| `--no-stats` | — | Opt out of anonymous performance stats for this run |

## Anonymous usage stats

To understand how Akiti TTS performs on real-world hardware, **each generation
sends a small, anonymous performance report** (from both the CLI and the GUI):

- your CPU model name, architecture, OS, Python version
- thread count and model variant (q4/q8)
- generation speed: audio length, generation time, real-time factor

**Never sent:** your input text, the generated audio, IP-linked identity, or any
personal data. The report describes *how fast the model ran*, never *what you
typed*.

### Why this helps

Akiti TTS runs on CPUs of every shape and size, and we can't test them all. These
anonymous reports let us build a public benchmark of real-time-factor across
different processors so we can:

- tune sensible defaults (thread counts, q4 vs q8) for common hardware,
- set realistic speed expectations in the docs, and
- prioritize optimization work where it actually matters.

The data is aggregate and anonymous — it directly improves the tool for everyone
running it.

### Opting out

Stats are sent by default. To opt out:

- **CLI, single run:** pass `--no-stats`.
- **Everywhere (CLI *and* GUI):** set the environment variable `AKITI_NO_STATS=1`.

```bash
python akiti_local.py --text "Meda wo ase." --no-stats
# or disable it everywhere, including the GUI:
export AKITI_NO_STATS=1
```

## License

CC BY-NC 4.0. Fine-tuned from VieNeu-TTS — please credit the upstream authors.
