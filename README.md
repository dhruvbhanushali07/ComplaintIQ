# ComplaintIQ — Customer Complaint Analysis & Resolution Prediction

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

Then open: http://localhost:5050

## Structure

- `generate_and_train.py` — Generates synthetic dataset + trains Random Forest model
- `app.py`               — Flask web dashboard (all-in-one)
- `model.pkl`            — Pre-trained RandomForest (200 trees, 76.7% accuracy)
- `stats.json`           — Pre-computed analytics for dashboard
- `complaints_dataset.csv` — 3,000-row synthetic dataset

## Re-train Model

```bash
python generate_and_train.py
python app.py
```

## Features

- **Overview Tab**: KPIs, priority distribution, issue frequency, monthly volume
- **Analysis Tab**: Resolution rates by category & priority, avg resolution time
- **Model Tab**: Confusion matrix, feature importance, classification report
- **Predict Tab**: Input complaint details → get resolution prediction + confidence
