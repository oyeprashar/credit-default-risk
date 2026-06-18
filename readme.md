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