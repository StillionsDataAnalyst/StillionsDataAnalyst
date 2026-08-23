# Predictive Maintenance Classification Demo

This portfolio project demonstrates an end-to-end machine-learning workflow for identifying elevated equipment failure risk within the next 14 days.

## Why this model exists

Manufacturing teams often have operational signals such as downtime, cycle-time variation, alarms, maintenance activity, OEE, and JPH attainment. The goal of this demonstration is to show how those signals can be prepared, modeled, compared, and evaluated before a predictive-maintenance system is considered for operational use.

## Data privacy

The model uses **synthetic demonstration data only**. No confidential employer or plant data is included in this repository. The synthetic data is generated reproducibly inside `train_model.py`.

## Features

- downtime hours in the previous 30 days
- average cycle time
- cycle-time variation
- alarm count in the previous 30 days
- maintenance events in the previous 90 days
- days since preventive maintenance
- OEE
- JPH attainment

Target: `failure_risk_14d`, a binary indicator representing elevated failure risk in the next 14 days.

## Models compared

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models use a stratified 75/25 train-test split with a fixed random seed for reproducibility. Class weighting is used because the positive class represents a smaller share of the dataset.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.727 | 0.305 | **0.644** | **0.414** | **0.760** |
| Decision Tree | 0.653 | 0.219 | 0.511 | 0.307 | 0.624 |
| Random Forest | **0.833** | **0.424** | 0.311 | 0.359 | 0.697 |

### Selected model: Logistic Regression

Logistic Regression is selected for this demonstration because it produced the strongest **recall** and **ROC-AUC** among the compared models. In a maintenance-screening use case, failing to flag a truly elevated-risk condition may be more costly than investigating a false alert, so recall is emphasized over accuracy alone.

This comparison also demonstrates why accuracy should not be the only model-selection metric. Random Forest had the highest accuracy, but its lower recall means it missed more of the positive-risk cases in this test.

## Run the project

```bash
pip install -r requirements.txt
python train_model.py
```

The script writes a generated demonstration dataset and `metrics.json` into an `outputs` folder.

## Responsible-use boundaries

This is a portfolio demonstration, not a production maintenance system. Before real deployment, the workflow would need plant-specific data validation, leakage checks, time-aware validation, threshold tuning, maintenance-expert review, drift monitoring, documented escalation rules, and verification that predictions improve operational outcomes without creating unsafe reliance on automation.

## Next steps

A future version can replace the synthetic generator with a secure preprocessing pipeline for approved operational data, add time-based cross-validation, tune alert thresholds based on the cost of missed failures versus false alarms, add feature-importance and calibration analysis, and connect predictions to a dashboard or FactoryAssist workflow.
