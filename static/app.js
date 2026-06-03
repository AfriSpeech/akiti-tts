"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  text: $("text"), gen: $("generate"), error: $("error"),
  status: $("status"), statusText: $("status-text"),
  charcount: $("charcount"), model: $("model"),
  temp: $("temperature"), tempVal: $("temp-val"),
  rep: $("rep"), repVal: $("rep-val"),
  clips: $("clips"), empty: $("empty"), clear: $("clear"),
};

let ready = false;
let clipCount = 0;

// --- Model status polling --------------------------------------------------
async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    if (s.error) {
      setStatus("error", "Model error — see terminal");
      els.error.textContent = s.error;
      els.error.hidden = false;
    } else if (s.ready) {
      ready = true;
      setStatus("ready", `Ready · ${s.model} · ${s.threads} threads`);
      els.gen.disabled = false;
      return; // stop polling
    } else {
      setStatus("loading", "Loading model…");
    }
  } catch (_) {
    setStatus("loading", "Connecting…");
  }
  setTimeout(pollStatus, 1000);
}

function setStatus(cls, text) {
  els.status.className = "status " + cls;
  els.statusText.textContent = text;
}

// --- Generate --------------------------------------------------------------
async function generate() {
  const text = els.text.value.trim();
  els.error.hidden = true;
  if (!text) { showError("Please enter some Twi text."); return; }
  if (!ready) return;

  setBusy(true);
  try {
    const res = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        model: els.model.value,
        temperature: parseFloat(els.temp.value),
        rep_penalty: parseFloat(els.rep.value),
      }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Generation failed (${res.status})`);
    }

    const blob = await res.blob();
    addClip({
      text,
      url: URL.createObjectURL(blob),
      gen: res.headers.get("X-Gen-Seconds"),
      audio: res.headers.get("X-Audio-Seconds"),
      rtf: res.headers.get("X-RTF"),
      model: res.headers.get("X-Model"),
    });
  } catch (e) {
    showError(e.message);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  els.gen.disabled = busy || !ready;
  els.gen.classList.toggle("busy", busy);
  els.gen.querySelector(".btn-label").textContent = busy ? "Generating…" : "Generate";
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = false;
}

// --- Clip library ----------------------------------------------------------
function addClip({ text, url, gen, audio, rtf, model }) {
  els.empty.hidden = true;
  els.clear.hidden = false;
  clipCount += 1;

  const li = document.createElement("li");
  li.className = "clip";

  const p = document.createElement("p");
  p.className = "clip-text";
  p.textContent = text;

  const player = document.createElement("audio");
  player.controls = true;
  player.autoplay = true;
  player.src = url;

  const meta = document.createElement("div");
  meta.className = "clip-meta";

  const stats = document.createElement("span");
  stats.className = "stats";
  stats.textContent = `${audio}s audio · ${model} · gen ${gen}s · RTF ${rtf}x`;

  const dl = document.createElement("a");
  dl.className = "dl";
  dl.href = url;
  dl.download = filenameFor(text, clipCount);
  dl.textContent = "⬇ Download";

  meta.append(stats, dl);
  li.append(p, player, meta);
  els.clips.prepend(li);
}

function filenameFor(text, n) {
  const slug = text.toLowerCase().replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "").slice(0, 32) || "akiti";
  return `akiti-${String(n).padStart(2, "0")}-${slug}.wav`;
}

function clearClips() {
  els.clips.innerHTML = "";
  els.clear.hidden = true;
  els.empty.hidden = false;
}

// --- Wiring ----------------------------------------------------------------
function updateCharCount() {
  const n = els.text.value.length;
  els.charcount.textContent = `${n} character${n === 1 ? "" : "s"}`;
}

els.gen.addEventListener("click", generate);
els.clear.addEventListener("click", clearClips);
els.text.addEventListener("input", updateCharCount);
els.temp.addEventListener("input", () => els.tempVal.textContent = parseFloat(els.temp.value).toFixed(2));
els.rep.addEventListener("input", () => els.repVal.textContent = parseFloat(els.rep.value).toFixed(2));

// Ctrl/Cmd + Enter to generate.
els.text.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); generate(); }
});

updateCharCount();
pollStatus();
