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

**Part 1 — Core algorithms.** Notebooks 00–07 cover the ideas every ML practitioner needs.

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

**Part 2 — Practical ML on a real, messy dataset.** Notebooks 08–11 follow one project: predicting which customers of a subscription business will cancel ("churn").

| # | Notebook | You'll learn |
|---|---|---|
| 08 | [Data Preparation & EDA](08_data_preparation_and_eda.ipynb) | spotting data problems · duplicates, text-as-numbers, messy categories, impossible values · **leaky columns** · EDA plots · **missing values** (MCAR/MAR/MNAR, imputation) · outliers (IQR, z-score, log) · one-hot / ordinal / **target encoding** · scaling · feature engineering · leak-proof **Pipelines** |
| 09 | [An End-to-End Project](09_end_to_end_project.ipynb) | framing the problem in **business costs** · baselines · comparing models fairly · tuning with random search, `RandomizedSearchCV` and **Optuna** · error analysis · **calibration** · choosing a threshold from costs · bootstrap confidence intervals · saving and serving a model · monitoring |
| 10 | [Explaining Models](10_explaining_models.ipynb) | odds ratios · **permutation importance** · **PDP & ICE** curves · **Shapley values** computed exactly from scratch · **SHAP** (waterfall, beeswarm, dependence) · **LIME** from scratch · counterfactuals · pitfalls (correlation, causation, leakage) |
| 11 | [Gradient Boosting in Practice](11_gradient_boosting_in_practice.ipynb) | the **second-order maths** of XGBoost · **histogram binning** · a mini-XGBoost in NumPy · **XGBoost vs LightGBM vs CatBoost** · early stopping · categorical & missing values · the hyperparameters that matter · Optuna tuning · class imbalance · **monotonic constraints** |

**Part 3 — Time series and the maths behind ML.** Notebooks 12–14 can be read in any order after Part 1. Notebooks 13 and 14 make a good refresher whenever an earlier notebook's maths feels shaky.

| # | Notebook | You'll learn |
|---|---|---|
| 12 | [Time Series Forecasting](12_time_series.ipynb) | trend/seasonality decomposition (moving averages, STL) · stationarity & differencing · **ACF/PACF** · MAE/RMSE/MAPE/**MASE** · naive & seasonal-naive baselines · exponential smoothing & Holt-Winters · **AR, ARIMA, SARIMA** · **Fourier terms** for multiple seasonalities · **lag features** with ML (and the rolling-window leak) · recursive forecasting · **walk-forward validation** · prediction intervals |
| 13 | [Probability & Statistics](13_probability_and_statistics.ipynb) | probability by simulation · **Bayes' rule** & base rates · distributions · **LLN & CLT** · **MLE/MAP** (and why MSE and log-loss are what they are) · bias of estimators · **confidence intervals** & the **bootstrap** · permutation tests, t-tests, p-values · **A/B testing**, power & sample size · the **peeking** problem · **multiple testing** (Bonferroni, Benjamini–Hochberg) · **Bayesian A/B** · Simpson's paradox |
| 14 | [Linear Algebra & Calculus](14_linear_algebra_and_calculus.ipynb) | vectors, norms, **cosine similarity** · matrices as transformations · determinants · solving systems & **conditioning** · **eigenvectors** (power iteration, PageRank) · **SVD** & low-rank image compression · derivatives three ways (numerical, SymPy, autograd) · **gradients** · **chain rule** · Jacobians & **Hessians** · Taylor approximations · convexity & saddle points · how **curvature** limits the learning rate · Newton's method · **Lagrange multipliers** (PCA as a constrained problem) |

**Part 4 — Deep learning: vision, sequences and language.** Notebooks 15–17 build on notebook 07 (neural networks) and are best read in order.

| # | Notebook | You'll learn |
|---|---|---|
| 15 | [Convolutional Neural Networks](15_convolutional_neural_networks.ipynb) | why MLPs struggle with images · **convolution** three ways and classic filters on a photo · padding, stride, channels, pooling · receptive fields · locality, weight sharing and **translation equivariance** · CNN vs MLP, including a **shifted-objects test** · learned filters and feature maps · a trained CNN run in plain NumPy · **data augmentation** · BatchNorm · **ResNet** skip connections · transfer learning · architecture milestones |
| 16 | [Sequence Models & Transformers](16_sequence_models_and_transformers.ipynb) | where bag-of-words fails · **RNNs** with hand-written **backprop through time** · **vanishing gradients** · **LSTM/GRU** gates (and why their initialisation matters) · a long-memory test · **attention** three ways · why divide by √d · causal masks · **multi-head attention** · **positional encodings** · a **transformer** built from parts, with attention maps · O(n²) cost |
| 17 | [NLP & Large Language Models](17_nlp_and_llms.ipynb) | **BPE tokenisation** from scratch and with `tokenizers` · TF-IDF and **BM25** search · word embeddings (PPMI + SVD) · n-gram language models and **perplexity** · a **tiny GPT** trained from scratch · decoding (greedy, temperature, **top-k, top-p**) · scaling laws and the pretraining → SFT → RLHF pipeline · **KV cache** · **LoRA** fine-tuning · **quantisation** · **RAG** with retrieval evaluation and query rewriting · evaluating LLMs |

**Part 5 — Specialised topics and ML in production.** Notebooks 18–21 can be read in any order after Parts 1 and 4. Notebook 22 is the finale and builds on 08–09.

| # | Notebook | You'll learn |
|---|---|---|
| 18 | [Reinforcement Learning](18_reinforcement_learning.ipynb) | **bandits** (ε-greedy, UCB, Thompson sampling) · MDPs and the **Bellman equations** · **value and policy iteration** · Monte Carlo vs **TD learning** · **Q-learning vs SARSA** on the cliff · **REINFORCE** on a CartPole simulator written from scratch · DQN, actor-critic, PPO and **RLHF** · reward hacking |
| 19 | [Recommender Systems](19_recommender_systems.ipynb) | the long tail and **missing-not-at-random** data · time-based splits · **precision/recall/NDCG@k** and coverage · biases and popularity baselines · **item–item CF** · **matrix factorisation** (SGD, ALS, PyTorch) · why low RMSE ≠ good rankings · **implicit feedback & BPR** · **cold start** · **two-tower retrieval** with in-batch negatives |
| 20 | [Graph Neural Networks](20_graph_neural_networks.ipynb) | adjacency matrices, PageRank, clustering · **label propagation** · **message passing** three ways · a **GCN** from scratch for semi-supervised node classification · **homophily vs heterophily** · **over-smoothing** · **link prediction** · graph classification and the **limits of message passing** · GraphSAGE, GAT, GIN |
| 21 | [Generative Models](21_generative_models.ipynb) | **KDE** and **GMMs** · **autoencoders** · **VAEs** (reparameterisation trick, ELBO, latent-space interpolation) · **GANs** and mode collapse · **diffusion (DDPM)** built from scratch · evaluating generative models · the modern landscape |
| 22 | [ML in Production](22_ml_in_production.ipynb) | a **model registry** · request schemas and **batch validation** · a real **FastAPI** service with latency percentiles and batching · **training–serving skew** · **drift monitoring** (PSI, KS) on a simulated 16-week dashboard · **data vs concept drift** · shadow, canary and **A/B rollout** (and choosing the right metric) · retraining · **behavioural tests** for models · a readiness checklist |

Take them in order: each notebook builds on the ones before it. Gradient descent (01) and evaluation (03) are used everywhere afterwards. Part 2 assumes Part 1, and its four notebooks share the same dataset and are best read in order. Part 3 can be read in any order after Part 1. Part 4 builds on notebook 07. Part 5 can be read in any order, with notebook 22 last.

## Setup

```bash
pip install -r requirements.txt
jupyter lab            # or: jupyter notebook, or open the folder in VS Code
```

You don't need to download anything. Part 1 uses datasets that ship with scikit-learn. Part 2 uses `data/customer_churn.csv`, a synthetic but realistically messy table generated by `data/make_churn_data.py`. Because the true churn formula is known, notebooks 10 and 11 can check whether models and explanations find the real drivers. The cleaning code that notebook 08 develops lives in `churn_utils.py`, which notebooks 09–11 import. Notebooks 16–17 use `text_data.py`. It provides a text corpus built from Python's own standard-library documentation (cached in `data/python_docs_corpus.txt`) and a generated set of product reviews.
The notebooks are saved **with their outputs**, so you can read them on GitHub without running anything.
Most notebooks run top to bottom in about a minute on a laptop CPU. Notebook 17 trains a small GPT and takes about 3 minutes. Nothing needs a GPU.

You can also open them in **Google Colab**: upload the `.ipynb` file. NumPy, scikit-learn and PyTorch are pre-installed there. For notebooks 08–11, also upload `churn_utils.py` and the `data/` folder, and run `!pip install shap optuna catboost`. Notebook 12 needs `statsmodels`, which is pre-installed on Colab. For notebooks 16–17, also upload `text_data.py` and run `!pip install tokenizers`. Notebook 22 needs `fastapi` and `httpx`.

## Where to go next

The course is complete: 23 notebooks, from a dot product to a monitored production system. Some natural extensions:
- **Real datasets:** rerun notebooks 08–11 on a public dataset (e.g. Telco churn from Kaggle) and notebooks 15–17 on MNIST/CIFAR-10 or a larger text corpus.
- **Bigger models:** fine-tune a pretrained vision model (`torchvision`) or language model (Hugging Face `transformers`), using the transfer-learning and LoRA ideas from notebooks 15 and 17.
- **Interview practice:** work through the `*_Interview_Notes.md` files. Every notebook links to the matching sections.

## Rebuilding

The notebooks are generated from plain-text sources in `notebooks/_src/`, where `%%md` starts a markdown cell and `%%code` starts a code cell.
To regenerate and re-run them:

```bash
python notebooks/_src/build.py notebooks/_src notebooks --execute
```
