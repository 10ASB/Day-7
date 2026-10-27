"""Day 7 lab: Decision Trees and Random Forest."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


def report(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred, test_pred = model.predict(X_train), model.predict(X_test)
    print(f"\n{'=' * 60}\n{name}")
    print(f"Train accuracy: {accuracy_score(y_train, train_pred):.3f}")
    print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.3f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, test_pred))
    print(classification_report(y_test, test_pred, zero_division=0))
    return model


def main():
    feature_names = [f"feature_{i}" for i in range(1, 7)]
    X, y = make_classification(n_samples=800, n_features=6, n_informative=4,
        n_redundant=1, n_classes=2, weights=[0.55, 0.45],
        class_sep=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    candidates = [
        ("Decision Tree — Gini, unrestricted",
         DecisionTreeClassifier(criterion="gini", random_state=42)),
        ("Decision Tree — Entropy, depth 3",
         DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)),
        ("Decision Tree — Gini, depth 5",
         DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)),
        ("Random Forest — 200 trees",
         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]
    results, fitted = [], {}
    for name, model in candidates:
        fitted[name] = report(name, model, X_train, X_test, y_train, y_test)
        results.append((name, accuracy_score(y_train, fitted[name].predict(X_train)),
                        accuracy_score(y_test, fitted[name].predict(X_test))))

    print("\nMODEL SUMMARY")
    print(f"{'Model':43s} {'Train':>8s} {'Test':>8s}")
    for name, train_score, test_score in results:
        print(f"{name:43s} {train_score:8.3f} {test_score:8.3f}")

    print("\nDecision Tree tuning (5-fold CV)")
    tree_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
        {"criterion": ["gini", "entropy"], "max_depth": [None, 3, 5, 8, 12],
         "min_samples_leaf": [1, 2, 5]}, cv=5, scoring="f1", n_jobs=-1)
    tree_search.fit(X_train, y_train)
    print("Best parameters:", tree_search.best_params_)
    print(f"Best CV F1: {tree_search.best_score_:.3f}")
    print(f"Tuned tree test accuracy: {accuracy_score(y_test, tree_search.predict(X_test)):.3f}")

    print("\nRandom Forest tuning (5-fold CV)")
    forest_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
         "max_features": ["sqrt", None], "min_samples_leaf": [1, 2]},
        cv=5, scoring="f1", n_jobs=-1)
    forest_search.fit(X_train, y_train)
    print("Best parameters:", forest_search.best_params_)
    print(f"Best CV F1: {forest_search.best_score_:.3f}")
    print(f"Tuned forest test accuracy: {accuracy_score(y_test, forest_search.predict(X_test)):.3f}")

    forest = fitted["Random Forest — 200 trees"]
    order = np.argsort(forest.feature_importances_)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[order], forest.feature_importances_[order])
    plt.xlabel("Mean decrease in impurity (relative importance)")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("day07_feature_importance.png", dpi=150)
    plt.show()
    print("Saved plot: day07_feature_importance.png")


if __name__ == "__main__":
    main()
