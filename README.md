# Day 7 — Decision Trees, Entropy, Random Forest & Feature Importance

**AI/ML Internship · Week 2 · Day 7**

This repository is organized around the three actual Day 7 notebooks.

## Tasks

### Task 1 — Entropy and Information Gain
- Define an entropy function.
- Calculate parent entropy and the weighted entropy after a split.
- Calculate information gain and verify the hand calculation.

### Task 2 — Decision Tree depth and overfitting
- Load the Iris dataset and split into train/test data.
- Train Decision Trees with `max_depth=20` and `max_depth=4`.
- Compare training and testing accuracy.
- Visualize both trees with `plot_tree`.

### Task 3 — Random Forest and feature importance
- Train a Random Forest with out-of-bag scoring.
- Inspect Gini feature importance and permutation importance.
- Compare a depth-4 Decision Tree against a Random Forest.

## Files

- [`TASK1.ipynb`](TASK1.ipynb) — entropy and information gain.
- [`TASK2.ipynb`](TASK2.ipynb) — decision tree visualization and overfitting.
- [`TASK3.ipynb`](TASK3.ipynb) — Random Forest, feature importance and comparison.
- `day07_trees_random_forest.py` — supplementary broader experiment with tree/forest tuning.
- `day07_lab_guide.md` — task notes and checklist.

## Setup

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

Open the notebooks in Jupyter or Google Colab and run each notebook from top to bottom.

## Important observation

In Task 1, `"Yes"` and `"YES"` are different string labels in Python, so the entropy function counts them as separate classes. The notebook preserves this input as supplied. You can experiment with consistent capitalization as a follow-up, but record it as a separate experiment.

The Iris notebooks use fixed random seeds, but scores can differ if you change the split, parameters or library version.
