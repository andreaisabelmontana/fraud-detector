# Fraud Detector — Threshold Dashboard

A portfolio-quality interactive dashboard for the classic imbalanced-binary-classification
problem: credit-card fraud detection.

> **Live:** https://andreaisabelmontana.github.io/fraud-detector/

## What it does

Drag the threshold slider and watch:

- the **confusion matrix** (TP / FP / FN / TN) re-tally
- **precision, recall, F1, accuracy** update
- a "you are here" dot move along the **ROC curve**
- every point in the **scatter plot** pick up an outline if it's now flagged
- the **live transaction stream** at the bottom re-classify each row

The dataset is synthetic — 1,000 transactions, ~1.5% fraud, deterministic seed so the
dashboard is identical for everyone. The "model" is a calibrated score function with
realistic overlap, so different thresholds genuinely change the trade-off.

## Tech

Plain HTML + CSS + JS. No frameworks, no build step.

```
fraud-detector/
├── index.html
├── style.css
├── favicon.svg
├── src/
│   ├── data.js        seeded PRNG + procedural transaction generator + metric helpers
│   └── app.js         UI wiring, canvas drawing (ROC + scatter), live stream
└── .github/workflows/deploy.yml
```

## Credits

Inspired by **Geethika**'s
[`Geethika2506/Creditcard-fraud-detection-Machine-Learning-final-project`](https://github.com/Geethika2506/Creditcard-fraud-detection-Machine-Learning-final-project) —
a Jupyter notebook doing EDA + model selection on the public Kaggle credit-card fraud dataset.
This dashboard is an independent portfolio piece by Andrea Montana (IE BCSAI, Fall 2025) —
all code, prose, and the synthetic transaction generator written from scratch.

No real customer data is used; every transaction on the page is procedurally generated in
your browser.
