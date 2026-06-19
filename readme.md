# Results
### XGBoost wins!

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.9194   | 0.5188    | 0.0139 | 0.0271 | 0.7447  |
| Random Forest       | 0.9194   | 0.8750    | 0.0014 | 0.0028 | 0.7118  |
| <span style="color:#e05c5c">**XGBoost**</span> | <span style="color:#e05c5c">**0.9197**</span> | <span style="color:#e05c5c">**0.5813**</span> | <span style="color:#e05c5c">**0.0187**</span> | <span style="color:#e05c5c">**0.0363**</span> | <span style="color:#e05c5c">**0.7564**</span> |
| Neural Network      | 0.8853   | 0.1815    | 0.1200 | 0.1445 | 0.6100  |



# Balanced XGBoost
| Metric    | Original XGBoost | Balanced XGBoost |
|-----------|------------------|------------------|
| Accuracy  | 0.9197           | 0.7135           |
| Precision | 0.5810           | 0.1700           |
| Recall    | 0.0187           | 0.6580           |
| F1        | 0.0360           | 0.2700           |
| ROC-AUC   | 0.7564           | 0.7533           |

Using scale_pos_weight (neg_example / pos_examples) dramatically increased recall from 1.9% to 65.8%, meaning the model found many more risky customers.
This came at the cost of lower precision and accuracy because the model produced more false positives.
ROC-AUC remained roughly unchanged, indicating the ranking quality of the model was similar.

# Balanced XGBoost Threshold Experiment

| Threshold | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| 0.1       | 0.083     | 0.996  | 0.154 |
| 0.2       | 0.095     | 0.962  | 0.173 |
| 0.3       | 0.114     | 0.891  | 0.203 |
| 0.4       | 0.140     | 0.793  | 0.238 |
| 0.5       | 0.170     | 0.658  | 0.270 |
| 0.6       | 0.209     | 0.492  | 0.293 |
| **0.7**   | **0.272** | **0.326** | **0.297** |