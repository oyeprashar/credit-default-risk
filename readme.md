# Logistic Regression Results

**Accuracy:** 0.9194

## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.92 | 1.00 | 0.96 | 56,538 |
| 1 | 0.52 | 0.01 | 0.03 | 4,965 |
| **Accuracy** | | | **0.92** | 61,503 |
| **Macro avg** | 0.72 | 0.51 | 0.49 | 61,503 |
| **Weighted avg** | 0.89 | 0.92 | 0.88 | 61,503 |

## Key Metrics (Class 1)

- **Precision:** 0.5188
- **Recall:** 0.0139

## Notes

The high overall accuracy is misleading due to class imbalance (class 0 dominates with ~92% of samples). The model essentially fails to identify class 1 cases, with a recall of only 1.4% — it's missing the vast majority of positive (default) cases.

---
# Random Forest — Model Evaluation
 
## Metrics Summary
 
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9194 |
| Precision | 0.8750 |
| Recall    | 0.0014 |
| F1 Score  | 0.0028 |
| ROC-AUC   | 0.7118 |
 
---
 
## Analysis
 
### Accuracy — 91.9%
Looks great on the surface, but misleading here. The dataset is heavily imbalanced — the model is likely predicting the majority class (non-default) almost every time.
 
### Precision — 87.5%
When the model *does* predict a positive (default), it's right 87.5% of the time. High, but practically useless given how rarely it predicts positive.
 
### Recall — 0.14%
The model catches almost **none** of the actual positives. Out of all real defaulters, it flags only 0.14%. This is the critical failure.
 
### F1 Score — 0.0028
Harmonic mean of Precision and Recall. Near zero because Recall is near zero. Confirms the model is failing on the minority class.
 
### ROC-AUC — 0.7118
The model has moderate ranking ability — 71.2% of the time it scores a true positive higher than a true negative. Decent, but not reflected in Recall because the classification threshold (0.5) is too high for this imbalanced dataset.
 
---