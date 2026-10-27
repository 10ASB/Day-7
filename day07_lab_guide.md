# Day 7 Lab Guide — Actual Notebook Tasks

## Task 1: Entropy and information gain
- Compute entropy for the parent labels.
- Calculate the weighted entropy of the left/right split.
- Calculate information gain by subtracting weighted entropy from parent entropy.
- Verify the result by hand and in code.
- Notice that `Yes` and `YES` count as different labels in Python.

## Task 2: Decision tree depth and overfitting
- Train the Iris Decision Tree at max_depth 20 and 4.
- Compare train accuracy with test accuracy.
- Inspect both plots produced by `plot_tree`.
- Explain how restricting depth affects complexity and potential overfitting.

## Task 3: Random Forest and feature importance
- Inspect train/test accuracy and out-of-bag score for the 100-tree forest.
- Compare Gini importance with permutation importance.
- Compare the accuracy of the depth-4 Decision Tree and 2,000-tree Random Forest on the second split.
- Note that feature importance is a model diagnostic, not proof of causation.

## Run
```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```
Open `TASK1.ipynb`, `TASK2.ipynb`, and `TASK3.ipynb` in Jupyter/Colab.

## Submission checklist
- [ ] Entropy and information gain printed and explained.
- [ ] Depth-20 and depth-4 trees visualized and compared.
- [ ] Random Forest OOB score and test score recorded.
- [ ] Gini and permutation importance compared.
- [ ] Decision Tree and Random Forest accuracies compared.
