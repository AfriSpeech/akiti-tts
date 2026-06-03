# Akiti TTS

Text-to-speech you can run **locally on your own CPU**. Akiti TTS currently
supports **Asante Twi**, with more languages planned. It is fine-tuned from
[VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B) and served
as a small quantized GGUF model, so it runs without a GPU.

- 🧠 Model: [afrispeech/Akiti-TTS](https://huggingface.co/afrispeech/Akiti-TTS)
- 🌐 Web demo: [michsethowusu/Akiti-TTS Space](https://huggingface.co/spaces/michsethowusu/Akiti-TTS)

The model weights download automatically from HuggingFace on first run and are
cached locally afterward.

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
# Synthesize to a WAV file
python akiti_local.py --text "Meda wo ase paa." --output out.wav

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

### REST API server

To call Akiti TTS from your own code or tools (curl, Python, n8n, etc.), run the
REST API server:

```bash
python akiti_api.py                       # loads q8, serves on http://127.0.0.1:8800
python akiti_api.py --model q4 --port 9000
```

The model loads once and stays warm. Long text is automatically split on
sentence boundaries, synthesized in parts, and stitched into a single clip.

**Endpoints**

| Method & path | Description |
|---|---|
| `GET /health` | Engine status (ready / loading / model / threads) |
| `GET /v1/voices` | Available voice presets |
| `POST /v1/synthesize` | Synthesize speech — returns a WAV (or JSON) |

**Synthesize — get a WAV back:**

```bash
curl -X POST http://127.0.0.1:8800/v1/synthesize \
     -H 'Content-Type: application/json' \
     -d '{"text": "Meda wo ase paa."}' --output out.wav
```

**Synthesize — get JSON back (base64 audio + metadata):**

```bash
curl 'http://127.0.0.1:8800/v1/synthesize?format=json' \
     -H 'Content-Type: application/json' \
     -d '{"text": "Akwaaba!", "model": "q8"}'
```

**Request body** (all optional except `text`):

| Field | Default | Description |
|---|---|---|
| `text` | — | Text to synthesize (**required**) |
| `model` | server's `--model` | `q4` or `q8` |
| `voice` | `none` | Voice preset name, or `none` (default speaker) |
| `temperature` | `0.4` | Higher = more varied |
| `top_p` | `0.8` | Nucleus sampling |
| `rep_penalty` | `1.3` | Higher reduces silences/repeats |

A binary WAV response includes `X-Gen-Seconds`, `X-Audio-Seconds`, `X-RTF`,
`X-Model`, and `X-Chunks` headers.

**Server options:** `--host 0.0.0.0` (expose on your network), `--port`,
`--model`, `--cors` (permissive CORS headers), and `--api-key KEY` (or the
`AKITI_API_KEY` environment variable) to require an
`Authorization: Bearer KEY` / `X-API-Key` header on requests.

### Options

| Flag | Default | Description |
|---|---|---|
| `--text` | — | Text to synthesize |
| `--model` | `q4` | `q4` (faster) or `q8` (better quality) |
| `--output` | `output.wav` | Output WAV path |
| `--temperature` | `0.4` | Higher = more varied |
| `--top-p` | `0.8` | Nucleus sampling |
| `--rep-penalty` | `1.3` | Higher reduces silences/repeats |
| `--threads` | all cores | CPU threads to use |
| `--no-stats` | — | Opt out of anonymous performance stats for this run |

## Anonymous usage stats

To understand how Akiti TTS performs on real-world hardware, **each generation
sends a small, anonymous performance report** (from the CLI, the Studio, and the API):

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
- **Everywhere (CLI *and* Studio):** set the environment variable `AKITI_NO_STATS=1`.

```bash
python akiti_local.py --text "Meda wo ase." --no-stats
# or disable it everywhere, including the Studio:
export AKITI_NO_STATS=1
```

## License

CC BY-NC 4.0. Fine-tuned from VieNeu-TTS — please credit the upstream authors.
