# Stress Detection in Speech using CatBoost

A machine learning pipeline for detecting stress levels from speech audio using the CREMA-D dataset. The model extracts MFCC and prosodic features from `.wav` files and trains a CatBoost regressor to predict a continuous stress score. Achieves **96% threshold-based accuracy**. Research paper submitted for publication.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML Model | CatBoost (CatBoostRegressor) |
| Audio Processing | librosa |
| Feature Engineering | MFCC, pitch (F0), energy, zero-crossing rate |
| Data Processing | NumPy, Pandas |
| Evaluation | Scikit-learn (MAE, MSE, RMSE, R²) |

---

## Dataset

- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) — 7,442 audio clips from 91 actors expressing 6 emotions (Anger, Disgust, Fear, Happy, Neutral, Sad) at varying intensities (LO / MD / HI / XX)
- **RAVDESS** — also used for supplementary data (referenced in `about.txt`)

---

## How It Works

1. **Feature Extraction** — for each `.wav` file:
   - 13 MFCC coefficients (mean across time frames)
   - Prosodic features: pitch mean, pitch std, energy, zero-crossing rate
   - Combined into a 17-dimensional feature vector

2. **Stress Label Generation** — rule-based mapping from emotion + intensity [weights sourced from a combination of research papers]:
   - High stress: Anger (0.8), Fear (0.8)
   - Medium stress: Sadness (0.7), Disgust (0.6)
   - Baseline: Neutral (0.5)
   - Intensity multiplier: LO=0.5, MD=0.75, HI=1.0
   - Final score clamped to [0, 1]

3. **Model Training**:
   - 80/20 train-test split
   - `StandardScaler` normalization
   - `CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, loss_function='RMSE')`

4. **Evaluation**:
   - Reports MAE, MSE, RMSE, R²
   - Threshold-based accuracy: prediction within ±0.35 of ground truth → **96% accuracy**

---

## Files

```
├── run.py                          # Main pipeline: extract → label → train → evaluate
├── correlation_calc.py             # Spearman/Pearson correlation analysis
├── emo_int_calc.py                 # Emotion-intensity mapping utilities
├── tabulatedVotes.csv              # CREMA-D crowd-sourced emotion vote data
├── Spearman_correlation_emotions.png
├── Pearson_correlation_emotions.png
└── about.txt                       # Dataset notes
```

---

## Getting Started

```bash
# Install dependencies
pip install catboost librosa scikit-learn numpy pandas

# Set your CREMA-D dataset path in run.py
CREMA_D_PATH = "./CREMA-D/AudioWAV"

# Run the pipeline
python run.py
```

> Requires Python 3.8+ and the CREMA-D AudioWAV folder downloaded separately.

---

## Results

| Metric | Value |
|---|---|
| Accuracy (±0.35 threshold) | **96%** |
| Loss Function | RMSE |
| Model | CatBoostRegressor |
| Iterations | 200 |

---

## Deployment

Not deployed. This is a research/ML pipeline — runs locally against the CREMA-D dataset.

---

## Assumptions & Notes

- Stress level is treated as a **regression** problem (continuous score 0–1), not binary classification
- The `about.txt` mentions RAVDESS was also used; the primary pipeline in `run.py` uses CREMA-D only — RAVDESS integration was not included here as the results did not significantly vary with the dataset.
- Paper submitted for publication; results are reproducible by running `run.py` with the CREMA-D dataset

---

## Author

**Shyam Narasimha Tadicharla**
GitHub: [Shyam-Narasimha-Tadicharla](https://github.com/Shyam-Narasimha-Tadicharla) | LinkedIn: [shyam-narasimha-tadicharla](https://www.linkedin.com/in/shyam-narasimha-tadicharla-750b6727a/)
