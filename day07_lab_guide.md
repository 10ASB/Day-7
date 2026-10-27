# Day 7 Lab Guide — Decision Trees and Random Forest

## Part A — Decision Trees
1. Run `day07_trees_random_forest.py`.
2. Compare Gini and entropy criteria.
3. Compare an unrestricted tree with trees limited to depths 3 and 5.
4. Record training and test accuracy. Which model has the largest gap?

## Part B — Bagging and Random Forest
1. Explain how bootstrap samples are generated.
2. Describe how Random Forest combines predictions from many trees.
3. Explain how random feature selection helps decorrelate trees.
4. Compare the forest and a single tree using the same split.

## Part C — Hyperparameter tuning
The script uses five-fold GridSearchCV on training data:
- Decision Tree: criterion, max_depth, min_samples_leaf.
- Random Forest: n_estimators, max_depth, max_features, min_samples_leaf.

Record the best parameters and cross-validation F1. Keep the held-out test set for final evaluation.

## Part D — Feature importance
Open `day07_feature_importance.png` after running the script. Identify the two highest-ranked features and explain why importance is predictive, not causal, evidence.

## Submission checklist
- [ ] Code runs successfully.
- [ ] Model comparison table includes train and test accuracy.
- [ ] Confusion matrices and classification reports reviewed.
- [ ] Best cross-validation settings recorded.
- [ ] Feature-importance plot saved.
- [ ] Reflection questions answered in your learning log.

## Reflection questions
1. What are Gini impurity, entropy, and information gain?
2. Why can a deep Decision Tree overfit?
3. How does bagging reduce variance?
4. How does Random Forest differ from plain bagging?
5. What are limitations of impurity-based feature importance?
