import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier


class ModelFactory:
    """
    Returns a sklearn Pipeline (StandardScaler → Estimator).
    Each pipeline is a fully self-contained, leak-free model unit.

    Supported model names:
        'majority'             – always predicts the majority class (baseline)
        'logistic_regression'  – L2-regularised logistic regression
        'svm_linear'           – Linear SVM
        'svm_rbf'              – RBF-kernel SVM
        'random_forest'        – Random Forest
        'xgboost'              – Gradient Boosting (sklearn, drop-in replacement)
    """

    # ── Default hyper-parameters ────────────────────────────────────────────
    _CONFIGS = {
        'majority': dict(
            estimator=DummyClassifier(strategy='most_frequent'),
            scale=False,
        ),
        'logistic_regression': dict(
            estimator=LogisticRegression(
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
                C=0.1,
                solver='lbfgs',
            ),
            scale=True,
        ),
        'svm_linear': dict(
            estimator=SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                class_weight='balanced',
                C=0.1,
            ),
            scale=True,
        ),
        'svm_rbf': dict(
            estimator=SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced',
                C=1.0,
                gamma='scale',
            ),
            scale=True,
        ),
        'random_forest': dict(
            estimator=RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
            ),
            scale=False,
        ),
        'xgboost': dict(
            estimator=GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
            scale=False,
        ),
    }

    @staticmethod
    def get_model(model_name: str) -> Pipeline:
        name = model_name.lower()
        if name not in ModelFactory._CONFIGS:
            supported = ', '.join(f"'{k}'" for k in ModelFactory._CONFIGS)
            raise ValueError(
                f"Model '{model_name}' is not supported. Choose from: {supported}."
            )

        cfg = ModelFactory._CONFIGS[name]
        steps = []
        if cfg['scale']:
            steps.append(('scaler', StandardScaler()))
        steps.append(('clf', cfg['estimator']))

        return Pipeline(steps)

    @staticmethod
    def all_model_names():
        return list(ModelFactory._CONFIGS.keys())