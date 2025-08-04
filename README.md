# Ridge Regression with Active Learning  
**Columbia+ Machine Learning – Module 3 Project**

This project implements **regularized least squares linear regression (Ridge Regression)** and an **Active Learning strategy** for efficient label acquisition. It is part of the Module 3 assignment for the Columbia+ Machine Learning course.

---

##  Assignment Overview

This project has two main components:

###  Part 1: Ridge Regression
Implement **Ridge Regression** to solve:
\[
\hat{w} = (X^\top X + \lambda I)^{-1} X^\top y
\]
Given training data `X_train` and `y_train`, and a regularization parameter `λ`, the function computes the weight vector `w`.

---

###  Part 2: Active Learning
Implement an **active learning procedure** that:
- Starts with a small labeled training set
- Iteratively selects 10 points from an unlabeled pool (`X_pool`) based on **uncertainty sampling**
- Simulates querying an oracle for labels
- Retrains the Ridge Regression model with new labeled data

---

##  Algorithm Workflow

1. **Initial Training**:
   - Train Ridge Regression model on a small labeled dataset.
2. **Active Learning Loop**:
   - Predict on unlabeled pool
   - Compute uncertainty (e.g., predictive variance)
   - Select the most uncertain data point
   - Simulate its label (using the current model or ground truth)
   - Add to labeled set and retrain
3. **Final Evaluation**:
   - Evaluate the model's performance after acquiring 10 more labeled points.

---

##  Dataset Files

The following files are used:

| File Name | Description |
|-----------|-------------|
| `X_train.csv` | Training input features (each row = one vector) |
| `y_train.csv` | Training output labels (each row = one value) |
| `X_test.csv`  | Test input features (same format as `X_train`) |

---

##  Visualizations

Visual aids included in the notebook to enhance understanding:

-  **Training Size Growth** over active learning iterations  
-  **Ridge Predictions vs Simulated Labels**  
-  **OLS vs Ridge vs Active Ridge** comparison  
-  **Prediction Uncertainty Distributions**  
-  **Selected Samples Highlighting** on test set  

---

##  Uncertainty Sampling Strategy

Uncertainty is measured using the **predictive variance**:
\[
\text{var}(x) = x^\top (X^\top X + \lambda I)^{-1} x
\]

The point with the **highest variance** is selected in each iteration.

##  Key Learnings
Implemented Ridge Regression from scratch using NumPy

Applied Active Learning to iteratively improve the model with fewer labels

Understood how uncertainty can guide label acquisition

Visualized model learning dynamics and uncertainty distributions

##  Technologies Used
Python 3.6+

NumPy, pandas

Matplotlib (for plotting)

Google Colab (runtime environment)

Gemini (for debugging, explanation, and optimization)

##  Reference
Columbia+ Machine Learning Course

Lecture 5: Active Learning

Lecture 3: Regularized Least Squares
