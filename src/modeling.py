from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def build_logistic_regression(penalty="l2", C=1.0):
    """
    Build a Logistic Regression model.

    Args:
        penalty: Regularization type ('l1' or 'l2')
        C: Inverse regularization strength

    Returns:
        Untrained Logistic Regression model
    """

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

    return model

def build_decision_tree(criterion='gini', max_depth=None, min_samples_split=2):
    """
    Build a Decision Tree classifier.

    Args:
        criterion: Split criterion.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples required to split.

    Returns:
        Untrained Decision Tree model.
    """

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    return model

def build_random_forest(n_estimators=200, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features="sqrt"):
    """
    Build a Random Forest classifier.

    Args:
        n_estimators: Number of trees.
        criterion: Split criterion.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples required to split.
        min_samples_leaf: Minimum samples in each leaf.
        max_features: Number of features considered at each split.

    Returns:
        Untrained Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    return model

def build_gradient_boosting(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=5, min_samples_split=10, subsample=0.8, max_features=None):
    """
    Build Gradient Boosting classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        criterion: Split criterion ('gini' or 'entropy')
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in a leaf
        max_features: Number of features considered at each split

    Returns:
        Untrained Gradient Boosting classifier model
    """

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42
    )

    return model

def build_xgboost(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8):
    """
    Build XGBoost classifier.
    """

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )

    return model

def build_lightgbm(n_estimators=200, learning_rate=0.05, max_depth=-1, num_leaves=31):
    """
    Build LightGBM classifier.
    """

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbosity=-1,
        n_jobs=-1,
        random_state=42
    )

    return model