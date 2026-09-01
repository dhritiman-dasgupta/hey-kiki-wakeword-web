# Hey KiKi — Browser Wake-Word Detector

A static web app that runs a trained **openWakeWord** model **entirely in your browser**
using [onnxruntime-web](https://onnxruntime.ai/) and the Web Audio API. Your microphone
audio never leaves your device.

**Live demo:** **<https://dhritiman-dasgupta.github.io/hey-kiki-wakeword-web/>** — say "hey kiki" into your mic; detection runs entirely in your browser, nothing is uploaded.

## What it does
- Loads three ONNX models in the browser: `melspectrogram` → `embedding` → `hey_kiki` (the wake-word classifier).
- Captures mic audio at 16 kHz, runs openWakeWord's exact streaming feature pipeline (ported to JS and verified to match the Python reference to < 1e-7), and shows a live confidence meter.
- Also scores an uploaded audio file (wav/mp3/m4a/aac) offline.

## Files
- `index.html` — UI
- `app.js` — mic capture + file scoring + glue
- `pipeline.js` — the openWakeWord feature pipeline (audio → mel → embedding → score)
- `models/` — `melspectrogram.onnx`, `embedding_model.onnx`, `hey_kiki.onnx`

## Run locally
```bash
# any static server works, e.g.:
python -m http.server 8080
# open http://localhost:8080
```

## Swap in a better model
This repo ships a small **demo** `hey_kiki.onnx` (low recall). Replace
`models/hey_kiki.onnx` with a production-trained model (same openWakeWord format) for real
accuracy — the app code is unchanged.

Built with [openWakeWord](https://github.com/dscripka/openWakeWord),
[Piper](https://github.com/rhasspy/piper-sample-generator), and
[F5-TTS](https://github.com/SWivid/F5-TTS).
