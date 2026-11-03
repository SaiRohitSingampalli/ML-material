# Learn ML from Scratch — Notebook Track

This folder is the **learning track**. It's written for someone seeing these ideas for the first time.
The `.md` notes and `*_from_scratch.py` files in the parent folder are the **interview track**: dense, complete and meant for revision.

Learn a topic here first, then use the notes to go deeper and prepare for interviews.
Every notebook ends with links to the matching sections.

## How each notebook is built

Every concept follows the same pattern:

1. **Intuition first:** what the algorithm is doing, in plain English, with an analogy.
2. **Pictures:** plots of the data, the decision boundary and the loss curve.
3. **The maths, one step at a time.** Every symbol is explained.
4. **The algorithm written three ways:**
   - 🐍 **Pure Python:** lists and loops only, so every step is visible.
   - 🔢 **NumPy:** the same maths with arrays. It's short, fast and reads like the formula.
   - 📦 **Library:** scikit-learn or PyTorch, which is what you'd use at work. Each one is checked against our own version.
5. **Pitfalls:** what goes wrong in practice.
6. **Exercises with worked solutions.**

## Learning path

| # | Notebook | You'll learn |
|---|---|---|
| 00 | [Start Here — ML & Python primer](00_start_here_ml_and_python_primer.ipynb) | what ML is · supervised/unsupervised/RL · the ML workflow · NumPy, pandas, matplotlib · your first model |
| 01 | [Linear Regression](01_linear_regression.ipynb) | MSE loss · normal equation · **gradient descent** (batch, SGD, mini-batch) · learning rate · feature scaling · polynomial features · over/underfitting · **Ridge & Lasso** |
| 02 | [Logistic Regression](02_logistic_regression.ipynb) | sigmoid · log-loss · decision boundaries · thresholds · **softmax** for multi-class |
| 03 | [Model Evaluation](03_model_evaluation.ipynb) | train/val/test · **data leakage** · cross-validation · **bias–variance** · learning curves · confusion matrix · precision/recall/F1 · **ROC-AUC**, PR curves · imbalanced data · hyperparameter search |
| 04 | [KNN, Naive Bayes & SVM](04_knn_naive_bayes_svm.ipynb) | nearest neighbours · curse of dimensionality · Bayes' rule · Gaussian & Multinomial NB (spam filter) · max-margin · hinge loss · **kernel trick** |
| 05 | [Decision Trees & Ensembles](05_decision_trees_and_ensembles.ipynb) | Gini/entropy · building a tree recursively · pruning · **Random Forest** · feature importance · **AdaBoost** · **gradient boosting** · XGBoost/LightGBM |
| 06 | [Unsupervised Learning](06_unsupervised_learning.ipynb) | **K-Means** (and ++) · elbow/silhouette · DBSCAN · hierarchical · GMM/EM · **PCA** (eigen, SVD, power iteration) · t-SNE · anomaly detection |
| 07 | [Neural Networks](07_neural_networks.ipynb) | neurons · XOR · activations · **backpropagation** · a tiny **autograd engine** in pure Python · NumPy MLP · gradient checking · SGD/momentum/Adam · initialisation · dropout · **PyTorch** |

Take them in order: each notebook builds on the ones before it. Gradient descent (01) and evaluation (03) are used everywhere afterwards.

## Setup

```bash
pip install -r requirements.txt
jupyter lab            # or: jupyter notebook, or open the folder in VS Code
```

All datasets ship with scikit-learn, so you don't need to download anything.
The notebooks are saved **with their outputs**, so you can read them on GitHub without running anything.
Every notebook runs top to bottom in under a minute on a laptop CPU.

You can also open them in **Google Colab**: upload the `.ipynb` file. NumPy, scikit-learn and PyTorch are pre-installed there.

## Planned next notebooks

These topics will extend the track and fill gaps in the current material:

- **08 · Data preparation & EDA:** missing values, outliers, categorical encoding, scikit-learn `Pipeline` / `ColumnTransformer`
- **09 · An end-to-end project:** from raw CSV to a saved model, with hyperparameter tuning (grid, random, Optuna)
- **10 · Explaining models:** SHAP values, partial dependence plots, LIME, permutation importance
- **11 · Gradient boosting in practice:** XGBoost, LightGBM and CatBoost, which settings matter, early stopping
- **12 · Time series:** time-ordered splits, lag features, baselines, ARIMA vs ML models
- **13 · Probability & statistics for ML:** distributions, MLE/MAP, hypothesis tests and A/B testing, with simulations
- **14 · Linear algebra & calculus for ML:** vectors, matrices, eigenvectors, SVD, gradients and the chain rule, made visual
- **15 · CNNs:** convolution from scratch, then PyTorch on images
- **16 · RNNs, LSTMs & attention:** sequence models and the transformer, built step by step
- **17 · NLP & LLMs:** tokenisation (BPE), embeddings, language models, fine-tuning, RAG
- **18 · Reinforcement learning:** bandits, MDPs, Q-learning, policy gradients
- **19 · Recommender systems:** collaborative filtering, matrix factorisation, two-tower retrieval
- **20 · Graph neural networks:** message passing and GCNs from scratch
- **21 · Generative models:** autoencoders, VAEs, GANs and diffusion
- **22 · ML in production:** serving, monitoring, drift and A/B rollout (pairs with `ML_System_Design_Notes.md`)

## Rebuilding

The notebooks are generated from plain-text sources in `notebooks/_src/`, where `%%md` starts a markdown cell and `%%code` starts a code cell.
To regenerate and re-run them:

```bash
python notebooks/_src/build.py notebooks/_src notebooks --execute
```
