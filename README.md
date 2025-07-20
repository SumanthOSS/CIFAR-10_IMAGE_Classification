# CIFAR-10_IMAGE_Classification

#  CIFAR-10 Image Classification using Traditional Machine Learning Algorithms

This project implements and evaluates **12 traditional machine learning algorithms** for **image classification** on the **CIFAR-10 dataset**. The goal is to analyze the performance of classical ML models on image data without deep learning, serving as a baseline comparison before deploying more complex architectures like CNNs.

---

##  Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of:
- 60,000 32x32 color images in 10 different classes
- 50,000 training images and 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Since traditional ML algorithms do not operate on raw images, each image was flattened into a 3072-length feature vector (32 x 32 x 3).



## ⚙️ Algorithms Implemented

The following 12 models were trained and evaluated:

| Model                            | Accuracy |
|----------------------------------|----------|
| XGBoost                          | 86%      |
| CatBoost                         | 86%      |
| Random Forest                    | 83%      |
| Gradient Boosting                | 83%      |
| AdaBoost                         | 81%      |
| Logistic Regression              | 81%      |
| Support Vector Machines (SVM)    | 81%      |
| Decision Trees                   | 75%      |
| Linear Discriminant Analysis     | 71%      |
| Naive Bayes                      | 69%      |
| K-Nearest Neighbors (KNN)        | 57%      |
| Quadratic Discriminant Analysis  | 51%      |

---

##  Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Macro and Weighted Averages

Evaluation was done on a test set of **100 samples** balanced across two classes for simplicity in binary analysis (can be extended to all 10 classes).









---

##  Tech Stack

- **Language**: Python
- **Libraries**:
  - `scikit-learn`
  - `xgboost`, `catboost`
  - `numpy`, `pandas`
  - `matplotlib`, `seaborn`
- **Development Environment**: Jupyter Notebook

---

##  Key Learnings

- Traditional ML models can perform reasonably well on image classification after feature extraction.
- XGBoost and CatBoost outperform all other models with **86% accuracy**.
- KNN and QDA perform poorly due to high dimensionality and lack of robust generalization in raw image space.

---

##  Future Work

- Apply **dimensionality reduction (PCA, t-SNE)** to improve KNN and LDA performance.
- Compare results with **CNN-based deep learning models**.
- Extend classification to all **10 classes** with multi-class metrics.











