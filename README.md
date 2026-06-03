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
| `--no-stats` | — | Don't send anonymous stats this run |
| `--reset-stats` | — | Re-ask the stats consent question |

## Anonymous usage stats (opt-in)

On the **first run** you'll be asked whether to share anonymous performance
metrics. If you agree, after each generation the script sends:

- your CPU model name, architecture, OS, Python version
- thread count, model variant (q4/q8)
- generation speed: audio length, generation time, real-time factor

**Never sent:** your input text, the generated audio, or any personal data.

This builds a public benchmark of how fast Akiti TTS runs across different CPUs.
You can decline (the default), opt out per run with `--no-stats`, or change your
mind with `--reset-stats`. Your choice is stored in
`~/.config/akiti-tts/config.json`.

## License

CC BY-NC 4.0. Fine-tuned from VieNeu-TTS — please credit the upstream authors.
