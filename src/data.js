// ============================================================
// In-browser visualisation stand-in for the real classifier.
//
// The genuine model is the Python scikit-learn pipeline in this repo
// (fraud/ + train.py): a class-weighted LogisticRegression scoring
// ROC-AUC 0.949 / PR-AUC 0.446 on held-out data, with a decision
// threshold chosen along the precision-recall curve.
//
// To keep this page self-contained and GitHub-Pages-safe (no backend,
// no model download), the dashboard replays a deterministic batch of
// ~1,000 synthetic transactions whose per-row score mirrors the real
// model's behaviour:
//   - heavy class imbalance (~1.5% fraud)
//   - a "model score" in [0,1] correlated with the true label but with
//     realistic overlap, so different thresholds yield genuinely
//     different confusion matrices
//
// Nothing here is real customer data. The seeded PRNG keeps the dataset
// identical across page loads so screenshots stay consistent.
// ============================================================

// ----- seeded PRNG --------------------------------------------------------
let _seed = 1729;
function srand() {
  _seed = (_seed * 1103515245 + 12345) & 0x7fffffff;
  return _seed / 0x7fffffff;
}
function pick(arr) { return arr[(srand() * arr.length) | 0]; }
function gauss(mean = 0, std = 1) {
  // Box–Muller; cheap normal sample
  const u = Math.max(1e-9, srand()), v = srand();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ----- categorical metadata ----------------------------------------------
const MERCHANTS = [
  { name: 'Quick Stop Convenience', cat: 'retail',        risk: 0.04 },
  { name: 'Daily Brew Coffee',      cat: 'food',          risk: 0.02 },
  { name: 'Sushi Garden',           cat: 'food',          risk: 0.03 },
  { name: 'Metro Grocer',           cat: 'grocery',       risk: 0.02 },
  { name: 'Apex Electronics',       cat: 'electronics',   risk: 0.18 },
  { name: 'Sapphire Watches',       cat: 'luxury',        risk: 0.28 },
  { name: 'Travelair Tickets',      cat: 'travel',        risk: 0.22 },
  { name: 'Cloudstay Hotels',       cat: 'travel',        risk: 0.20 },
  { name: 'Northwind Gas',          cat: 'fuel',          risk: 0.05 },
  { name: 'CityFit Gym',            cat: 'services',      risk: 0.02 },
  { name: 'PrimeStream Media',      cat: 'subscription',  risk: 0.04 },
  { name: 'Marble Pharmacy',        cat: 'pharmacy',      risk: 0.03 },
  { name: 'Iberia Books',           cat: 'retail',        risk: 0.03 },
  { name: 'Halcyon Jewelry',        cat: 'luxury',        risk: 0.32 },
  { name: 'Skylane Airlines',       cat: 'travel',        risk: 0.24 },
  { name: 'Plata Cosmetics',        cat: 'retail',        risk: 0.06 },
  { name: 'Vertex Hardware',        cat: 'hardware',      risk: 0.10 },
  { name: 'Onyx Auto Parts',        cat: 'auto',          risk: 0.12 },
  { name: 'Lumière Lighting',       cat: 'home',          risk: 0.06 },
  { name: 'Tide & Sun Resort',      cat: 'travel',        risk: 0.26 },
];

// ----- one transaction ----------------------------------------------------
function makeTx(id) {
  const m = pick(MERCHANTS);
  // Hour-of-day distribution: bimodal around lunch + evening, with a small
  // late-night tail — late hours raise the risk slightly.
  const hour = Math.floor(clamp(gauss(15, 5), 0, 23));
  const lateNight = hour >= 0 && hour <= 4;

  // Amount: log-normal, capped. Luxury and travel skew higher.
  let amount = Math.exp(gauss(2.7, 1.0));
  if (m.cat === 'luxury' || m.cat === 'travel') amount *= 4;
  amount = clamp(amount, 1.2, 4800);

  // Ground-truth fraud probability — combines merchant base risk,
  // amount magnitude, and time-of-day effect with a small Gaussian wobble.
  const sizeBoost  = Math.min(0.35, amount / 8000);
  const timeBoost  = lateNight ? 0.15 : 0;
  const noise      = gauss(0, 0.08);
  const trueP      = clamp(m.risk + sizeBoost + timeBoost + noise, 0.005, 0.97);

  // True label drawn from trueP, but scaled so the realised positive rate
  // sits around the ~1.5% we want.
  const isFraud    = srand() < trueP * 0.07;

  // Model score: starts from trueP, adds noise. The model is *good but not
  // perfect* — frauds tend to get high scores, legits low, with overlap.
  let score = clamp(trueP + gauss(0, 0.15), 0, 1);
  if (isFraud)        score = clamp(score + 0.25, 0, 1);
  else if (srand() < 0.04) score = clamp(score + 0.4, 0, 1);   // some legit mislabelled high

  // Two abstract "PCA-like" features for the scatter plot. We arrange them
  // so frauds tend to land in one quadrant — visually obvious cluster.
  const v1 = gauss(isFraud ? 1.5 : -0.2, 1.0);
  const v2 = gauss(isFraud ? 1.2 :  0.1, 1.1);

  return {
    id,
    merchant: m.name,
    category: m.cat,
    amount: +amount.toFixed(2),
    hour,
    isFraud,
    score: +score.toFixed(3),
    v1, v2,
  };
}

export function buildDataset(n = 1000) {
  _seed = 1729;
  const tx = [];
  for (let i = 0; i < n; i++) tx.push(makeTx(i + 1));
  return tx;
}

// ----- metrics ------------------------------------------------------------
export function confusion(tx, threshold) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const t of tx) {
    const predFraud = t.score >= threshold;
    if (predFraud && t.isFraud)        tp++;
    else if (predFraud && !t.isFraud)  fp++;
    else if (!predFraud && !t.isFraud) tn++;
    else                                fn++;
  }
  return { tp, fp, tn, fn };
}

export function metrics({ tp, fp, tn, fn }) {
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall    = tp + fn ? tp / (tp + fn) : 0;
  const f1        = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  const fpr       = fp + tn ? fp / (fp + tn) : 0;
  const acc       = (tp + tn) / (tp + fp + tn + fn || 1);
  return { precision, recall, f1, fpr, acc };
}

// ROC curve points by sweeping threshold from 1 -> 0 in 51 steps.
export function rocCurve(tx) {
  const pts = [];
  for (let i = 0; i <= 50; i++) {
    const th = 1 - i / 50;
    const c = confusion(tx, th);
    const tpr = c.tp + c.fn ? c.tp / (c.tp + c.fn) : 0;
    const fpr = c.fp + c.tn ? c.fp / (c.fp + c.tn) : 0;
    pts.push({ th, tpr, fpr });
  }
  return pts;
}
