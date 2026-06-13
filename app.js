import { WakeWord } from './pipeline.js';

const ort = window.ort;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const $ = (id) => document.getElementById(id);
const loadStatus = $('loadStatus'), micBtn = $('mic'), stopBtn = $('stop');
const scoreEl = $('score'), meter = $('meter'), meterBar = meter.firstElementChild;
const stateEl = $('state'), lastEl = $('last'), logEl = $('log');
const thEl = $('th'), thVal = $('thVal'), fileEl = $('file'), runBtn = $('run'), fileResult = $('fileResult');

let wk = null, sessions = null, listening = false;
let audioCtx = null, srcNode = null, proc = null, stream = null;
let pending = [], draining = false, lastFire = 0;

thEl.oninput = () => thVal.textContent = (+thEl.value).toFixed(2);

function log(msg) {
  const d = document.createElement('div');
  d.textContent = msg;
  logEl.prepend(d);
}

async function loadModels() {
  try {
    const opt = { executionProviders: ['wasm'] };
    const [mel, emb, ww] = await Promise.all([
      ort.InferenceSession.create('./models/melspectrogram.onnx', opt),
      ort.InferenceSession.create('./models/embedding_model.onnx', opt),
      ort.InferenceSession.create('./models/hey_kiki.onnx', opt),
    ]);
    sessions = { mel, emb, ww };
    wk = new WakeWord(ort, mel, emb, ww);
    loadStatus.innerHTML = '✅ Models loaded — ready. Click <b>Start listening</b> and say your wake word.';
    micBtn.disabled = false; runBtn.disabled = false;
  } catch (e) {
    loadStatus.textContent = '❌ Failed to load models: ' + e.message;
    console.error(e);
  }
}

function toInt16(float32, inRate) {
  // resample (linear) to 16k if needed, return Int16Array
  let data = float32;
  if (inRate !== 16000) {
    const ratio = inRate / 16000;
    const n = Math.floor(float32.length / ratio);
    data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * ratio, i0 = Math.floor(p), frac = p - i0;
      data[i] = (float32[i0] || 0) * (1 - frac) + (float32[i0 + 1] || 0) * frac;
    }
  }
  const out = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    let s = Math.max(-1, Math.min(1, data[i]));
    out[i] = Math.round(s * 32767);
  }
  return out;
}

async function drain() {
  if (draining) return;
  draining = true;
  while (listening && pending.length >= 1280) {
    const chunk = Int16Array.from(pending.splice(0, 1280));
    const score = await wk.pushChunk(chunk);
    if (score != null) updateScore(score);
  }
  draining = false;
}

function updateScore(score) {
  scoreEl.textContent = score.toFixed(3);
  meterBar.style.width = Math.min(100, score * 100) + '%';
  const th = +thEl.value;
  const hit = score >= th;
  meter.classList.toggle('hit', hit);
  if (hit && Date.now() - lastFire > 1500) {
    lastFire = Date.now();
    const t = new Date().toLocaleTimeString();
    log(`🔔 DETECTED  score=${score.toFixed(3)}  @ ${t}`);
    lastEl.textContent = `Last detection: ${t} (score ${score.toFixed(3)})`;
  }
}

async function startMic() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const inRate = audioCtx.sampleRate;
    srcNode = audioCtx.createMediaStreamSource(stream);
    proc = audioCtx.createScriptProcessor(4096, 1, 1);
    wk.reset();
    proc.onaudioprocess = (e) => {
      if (!listening) return;
      const f = e.inputBuffer.getChannelData(0);
      const i16 = toInt16(f, inRate);
      for (let i = 0; i < i16.length; i++) pending.push(i16[i]);
      drain();
    };
    srcNode.connect(proc);
    proc.connect(audioCtx.destination);
    listening = true;
    stateEl.textContent = 'listening'; stateEl.className = 'badge live';
    micBtn.disabled = true; stopBtn.disabled = false;
    log('🎙️ listening… (sample rate ' + inRate + ' Hz)');
  } catch (e) {
    log('mic error: ' + e.message);
  }
}

function stopMic() {
  listening = false;
  if (proc) proc.disconnect();
  if (srcNode) srcNode.disconnect();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioCtx) audioCtx.close();
  pending = [];
  stateEl.textContent = 'idle'; stateEl.className = 'badge idle';
  micBtn.disabled = false; stopBtn.disabled = true;
}

async function scoreFile() {
  const file = fileEl.files[0];
  if (!file) { fileResult.textContent = 'pick a file first'; return; }
  fileResult.textContent = 'decoding…';
  const buf = await file.arrayBuffer();
  const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await tmpCtx.decodeAudioData(buf);
  // resample to 16k mono via OfflineAudioContext
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  const pcm = toInt16(rendered.getChannelData(0), 16000);
  tmpCtx.close();

  const fwk = new WakeWord(ort, sessions.mel, sessions.emb, sessions.ww);
  let max = 0, peakT = 0, frames = 0;
  for (let i = 0; i + 1280 <= pcm.length; i += 1280) {
    const s = await fwk.pushChunk(pcm.subarray(i, i + 1280));
    if (s != null) { frames++; if (s > max) { max = s; peakT = (i / 16000); } }
  }
  const th = +thEl.value;
  fileResult.innerHTML = `peak confidence <b>${max.toFixed(4)}</b> at ${peakT.toFixed(2)}s — ` +
    (max >= th ? `<span style="color:#27c08a">DETECTED</span>` : `<span style="color:#f1c79a">not detected</span>`) +
    ` (threshold ${th.toFixed(2)}, ${frames} frames)`;
  log(`📄 ${file.name}: peak ${max.toFixed(4)}`);
}

micBtn.onclick = startMic;
stopBtn.onclick = stopMic;
runBtn.onclick = scoreFile;
loadModels();
