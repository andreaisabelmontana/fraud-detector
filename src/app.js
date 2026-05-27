// ============================================================
// Threshold dashboard wiring.
//
// One synthetic dataset, one threshold slider, and three views that all
// recompute together: the four-cell confusion matrix, the ROC curve with
// a "you-are-here" marker, and a scatter plot where every transaction's
// outline reflects what the current threshold predicts.
//
// The live transaction stream replays the dataset at a steady pace so
// the page is never static — each new row is classified using the
// current threshold, so dragging the slider also retroactively re-tags
// every visible row.
// ============================================================
import { buildDataset, confusion, metrics, rocCurve } from './data.js';

const tx = buildDataset(1000);
const fraudCount = tx.filter(t => t.isFraud).length;

// ----- DOM refs -----------------------------------------------------------
const thrEl   = document.getElementById('threshold');
const thrVal  = document.getElementById('thr-val');
const mPrec   = document.getElementById('m-prec');
const mRec    = document.getElementById('m-rec');
const mF1     = document.getElementById('m-f1');
const mAcc    = document.getElementById('m-acc');
const cmEls   = ['tn','fp','fn','tp'].reduce((acc, k) => (acc[k] = document.getElementById(`cm-${k}`), acc), {});
const aucEl   = document.getElementById('auc');
const rocCv   = document.getElementById('roc');
const scatCv  = document.getElementById('scatter');
const streamEl= document.getElementById('stream');
const btnStrm = document.getElementById('btn-stream');
const btnStep = document.getElementById('btn-step');

// ----- ROC pre-computed once ---------------------------------------------
const roc = rocCurve(tx);
// AUC via trapezoidal integration over the ROC (sorted by FPR).
const aucPts = [...roc].sort((a, b) => a.fpr - b.fpr);
let auc = 0;
for (let i = 1; i < aucPts.length; i++) {
  const a = aucPts[i - 1], b = aucPts[i];
  auc += (b.fpr - a.fpr) * (a.tpr + b.tpr) / 2;
}
aucEl.textContent = `AUC = ${auc.toFixed(3)}`;

// ----- Canvas setup -------------------------------------------------------
const dpr = Math.max(1, window.devicePixelRatio || 1);
function fit(cv, hCss = 300) {
  cv.width  = Math.floor(cv.clientWidth * dpr);
  cv.height = Math.floor(hCss * dpr);
  cv.style.height = hCss + 'px';
}
function fitAll() {
  fit(rocCv,   300);
  fit(scatCv,  380);
}
fitAll();
window.addEventListener('resize', () => { fitAll(); render(currentThreshold()); });

// ----- Recompute & render -------------------------------------------------
function currentThreshold() { return parseFloat(thrEl.value); }

function render(thr) {
  thrVal.textContent = thr.toFixed(2);
  const cm = confusion(tx, thr);
  const m  = metrics(cm);

  mPrec.textContent = (m.precision * 100).toFixed(1) + '%';
  mRec.textContent  = (m.recall    * 100).toFixed(1) + '%';
  mF1.textContent   = m.f1.toFixed(3);
  mAcc.textContent  = (m.acc       * 100).toFixed(1) + '%';

  cmEls.tn.textContent = cm.tn;
  cmEls.fp.textContent = cm.fp;
  cmEls.fn.textContent = cm.fn;
  cmEls.tp.textContent = cm.tp;

  drawROC(thr);
  drawScatter(thr);
  refreshStreamClassification(thr);
}

// ----- ROC drawing --------------------------------------------------------
function drawROC(thr) {
  const ctx = rocCv.getContext('2d');
  const W = rocCv.width, H = rocCv.height;
  ctx.clearRect(0, 0, W, H);
  const m = { l: 44 * dpr, r: 16 * dpr, t: 12 * dpr, b: 36 * dpr };
  const cw = W - m.l - m.r, ch = H - m.t - m.b;

  // Axes + grid
  ctx.strokeStyle = '#E6CBC0'; ctx.lineWidth = 1 * dpr;
  ctx.font = `${12 * dpr}px Inter, sans-serif`; ctx.fillStyle = '#78635A';
  for (let i = 0; i <= 4; i++) {
    const t = i / 4;
    const y = m.t + (1 - t) * ch;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(W - m.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(t.toFixed(2), m.l - 6 * dpr, y);
    const x = m.l + t * cw;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(t.toFixed(2), x, H - m.b + 6 * dpr);
  }

  // Diagonal (random classifier)
  ctx.strokeStyle = '#94A3B8'; ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.beginPath(); ctx.moveTo(m.l, m.t + ch); ctx.lineTo(W - m.r, m.t); ctx.stroke();
  ctx.setLineDash([]);

  // ROC line — sweep from highest threshold (origin) to lowest (top-right)
  ctx.strokeStyle = '#DC2626'; ctx.lineWidth = 2.6 * dpr;
  ctx.beginPath();
  for (let i = 0; i < roc.length; i++) {
    const p = roc[i];
    const x = m.l + p.fpr * cw, y = m.t + (1 - p.tpr) * ch;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // "You are here" marker — closest sampled threshold to thr
  let best = roc[0], bestD = Math.abs(roc[0].th - thr);
  for (const p of roc) {
    const d = Math.abs(p.th - thr);
    if (d < bestD) { best = p; bestD = d; }
  }
  ctx.fillStyle = '#DC2626';
  ctx.beginPath();
  ctx.arc(m.l + best.fpr * cw, m.t + (1 - best.tpr) * ch, 6 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2 * dpr;
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#78635A'; ctx.font = `600 ${11 * dpr}px Inter, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText('false positive rate', m.l + cw / 2, H - m.b + 18 * dpr);
  ctx.save();
  ctx.translate(12 * dpr, m.t + ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText('true positive rate', 0, 0);
  ctx.restore();
}

// ----- Scatter ------------------------------------------------------------
function drawScatter(thr) {
  const ctx = scatCv.getContext('2d');
  const W = scatCv.width, H = scatCv.height;
  ctx.clearRect(0, 0, W, H);
  const m = { l: 40 * dpr, r: 16 * dpr, t: 14 * dpr, b: 30 * dpr };
  const cw = W - m.l - m.r, ch = H - m.t - m.b;

  // Compute bounds once
  let v1mn = Infinity, v1mx = -Infinity, v2mn = Infinity, v2mx = -Infinity;
  for (const t of tx) {
    if (t.v1 < v1mn) v1mn = t.v1;  if (t.v1 > v1mx) v1mx = t.v1;
    if (t.v2 < v2mn) v2mn = t.v2;  if (t.v2 > v2mx) v2mx = t.v2;
  }
  const sx = (v) => m.l + (v - v1mn) / (v1mx - v1mn) * cw;
  const sy = (v) => m.t + (1 - (v - v2mn) / (v2mx - v2mn)) * ch;

  // Background grid
  ctx.strokeStyle = '#F0E0D8'; ctx.lineWidth = 1 * dpr;
  for (let i = 1; i <= 5; i++) {
    const x = m.l + (i / 6) * cw;
    ctx.beginPath(); ctx.moveTo(x, m.t); ctx.lineTo(x, m.t + ch); ctx.stroke();
    const y = m.t + (i / 6) * ch;
    ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(W - m.r, y); ctx.stroke();
  }

  // Plot legit first (so frauds sit on top)
  for (const t of tx) {
    const flagged = t.score >= thr;
    const x = sx(t.v1), y = sy(t.v2);
    const r = (t.isFraud ? 4.5 : 2.6) * dpr;
    const fill = t.isFraud ? 'rgba(220,38,38,0.85)' : 'rgba(148,163,184,0.45)';
    ctx.fillStyle = fill;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    if (flagged) {
      ctx.strokeStyle = t.isFraud ? '#16A34A' : '#F59E0B';
      ctx.lineWidth = 1.8 * dpr;
      ctx.stroke();
    }
  }

  // Legend
  const legend = [
    { c: 'rgba(220,38,38,0.85)', t: 'fraud (truth)' },
    { c: 'rgba(148,163,184,0.5)', t: 'legit (truth)' },
    { c: '#16A34A', t: 'caught (TP)', outline: true },
    { c: '#F59E0B', t: 'false alarm (FP)', outline: true },
  ];
  ctx.font = `600 ${11 * dpr}px Inter, sans-serif`;
  let lx = m.l + 6 * dpr, ly = m.t + 6 * dpr;
  for (const item of legend) {
    if (item.outline) {
      ctx.fillStyle = '#FFFFFF';
      ctx.beginPath(); ctx.arc(lx + 5 * dpr, ly + 5 * dpr, 5 * dpr, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = item.c; ctx.lineWidth = 2 * dpr;
      ctx.beginPath(); ctx.arc(lx + 5 * dpr, ly + 5 * dpr, 5 * dpr, 0, Math.PI * 2); ctx.stroke();
    } else {
      ctx.fillStyle = item.c;
      ctx.beginPath(); ctx.arc(lx + 5 * dpr, ly + 5 * dpr, 5 * dpr, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = '#3A2418'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(item.t, lx + 14 * dpr, ly + 5 * dpr);
    lx += ctx.measureText(item.t).width + 36 * dpr;
  }
}

// ----- Stream -------------------------------------------------------------
const STREAM_MAX = 24;
let streamIdx = 0;
let streaming = true;
let streamTimer = null;

function fmtTime(hour) {
  const min = (streamIdx * 17) % 60;
  return `${String(hour).padStart(2,'0')}:${String(min).padStart(2,'0')}`;
}

function pushRow() {
  const t = tx[streamIdx % tx.length];
  streamIdx++;
  const li = document.createElement('li');
  li.dataset.id = t.id;
  li.dataset.score = t.score;
  li.dataset.fraud = t.isFraud ? '1' : '0';
  li.innerHTML = `
    <span class="time">${fmtTime(t.hour)}</span>
    <span class="merchant">${t.merchant}</span>
    <span class="cat muted small">${t.category}</span>
    <span class="amount">$${t.amount.toFixed(2)}</span>
    <span class="score" title="model score"></span>
    <span class="badge"></span>
  `;
  streamEl.prepend(li);
  while (streamEl.children.length > STREAM_MAX) streamEl.removeChild(streamEl.lastChild);
  classifyRow(li, currentThreshold());
}

function classifyRow(li, thr) {
  const score = +li.dataset.score;
  const fraud = li.dataset.fraud === '1';
  const flagged = score >= thr;
  li.classList.remove('flagged', 'cleared', 'tp', 'fp', 'fn', 'tn');
  let kind = 'tn';
  if (flagged && fraud) kind = 'tp';
  else if (flagged && !fraud) kind = 'fp';
  else if (!flagged && fraud) kind = 'fn';
  li.classList.add(flagged ? 'flagged' : 'cleared', kind);
  li.querySelector('.score').textContent  = score.toFixed(3);
  const badge = li.querySelector('.badge');
  badge.className = 'badge ' + kind;
  badge.textContent = {
    tp: '✓ caught',
    fp: '⚠ false alarm',
    fn: '✕ missed',
    tn: 'cleared',
  }[kind];
}

function refreshStreamClassification(thr) {
  for (const li of streamEl.children) classifyRow(li, thr);
}

function startStream() {
  if (streamTimer) return;
  streaming = true;
  btnStrm.textContent = '⏸ Pause';
  streamTimer = setInterval(pushRow, 800);
  pushRow();
}
function stopStream() {
  streaming = false;
  btnStrm.textContent = '▶ Resume';
  clearInterval(streamTimer); streamTimer = null;
}

btnStrm.addEventListener('click', () => streaming ? stopStream() : startStream());
btnStep.addEventListener('click', () => { if (!streaming) pushRow(); });

// ----- Bind slider --------------------------------------------------------
thrEl.addEventListener('input', () => render(currentThreshold()));
render(currentThreshold());
startStream();

// Initial fill so the stream isn't empty for the first 5 seconds.
for (let i = 0; i < 6; i++) pushRow();

document.getElementById('yr').textContent = new Date().getFullYear();

// Friendly summary in the console so anyone opening devtools sees what's
// going on without having to read the README.
console.log(
  `%cFraud Detector%c — ${tx.length} synthetic transactions, ${fraudCount} fraud (${(fraudCount / tx.length * 100).toFixed(2)}%). ` +
  `Underlying scorer AUC = ${auc.toFixed(3)}.`,
  'color:#DC2626; font-weight:bold', 'color:inherit'
);
