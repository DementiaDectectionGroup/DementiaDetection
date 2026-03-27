import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


# ── Feature-reduction factory ────────────────────────────────────────────────
def make_reducer(mode: str, n_components: int = 30):
    """
    mode: 'none' | 'pca' | 'selectkbest'
    Returns a (name, transformer) tuple or None.
    """
    mode = mode.lower()
    if mode == 'pca':
        return ('reducer', PCA(n_components=n_components, random_state=42))
    if mode == 'selectkbest':
        return ('reducer', SelectKBest(f_classif, k=n_components))
    if mode == 'none':
        return None
    raise ValueError(f"Unknown reduction mode '{mode}'. Choose: none | pca | selectkbest")


# ── Hyper-parameter grids for tuning ────────────────────────────────────────
# Keys mirror pipeline step names: 'clf__<param>'
_PARAM_GRIDS = {
    'logistic_regression': {
        'clf__C':       [0.001, 0.01, 0.1, 1.0, 10.0],
        'clf__solver':  ['lbfgs', 'liblinear'],
    },
    'svm_linear': {
        'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
}


class ModelFactory:
    """
    Builds sklearn Pipelines: [scaler?] → [reducer?] → estimator.

    Supported model names:
        'majority'             – always predicts the majority class (baseline)
        'logistic_regression'  – L2-regularised logistic regression
        'svm_linear'           – Linear SVM
        'svm_rbf'              – RBF-kernel SVM
        'random_forest'        – Random Forest
        'gradient_boosting'    – sklearn GradientBoostingClassifier
    """

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
        'gradient_boosting': dict(
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
    def _build_pipeline(model_name: str, reduction: str = 'none',
                        n_components: int = 30) -> Pipeline:
        """Internal: assemble pipeline steps for a given model + reduction mode."""
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
        reducer = make_reducer(reduction, n_components)
        if reducer is not None:
            steps.append(reducer)
        steps.append(('clf', cfg['estimator']))
        return Pipeline(steps)

    @staticmethod
    def get_model(model_name: str, reduction: str = 'none',
                  n_components: int = 30) -> Pipeline:
        """Return a plain (untuned) pipeline."""
        return ModelFactory._build_pipeline(model_name, reduction, n_components)

    @staticmethod
    def get_tuned_model(model_name: str, reduction: str = 'none',
                        n_components: int = 30, cv: int = 3,
                        scoring: str = 'roc_auc') -> Pipeline:
        """
        Return a GridSearchCV-wrapped pipeline for models that have a param grid.
        Falls back to a plain pipeline for models without a defined grid.
        The inner CV (cv=3) runs entirely within the training fold → no leakage.
        """
        name = model_name.lower()
        pipeline = ModelFactory._build_pipeline(name, reduction, n_components)
        if name not in _PARAM_GRIDS:
            return pipeline  # no grid defined, return plain pipeline

        grid = GridSearchCV(
            pipeline,
            param_grid=_PARAM_GRIDS[name],
            cv=cv,
            scoring=scoring,
            refit=True,
            n_jobs=-1,
        )
        return grid

    @staticmethod
    def all_model_names():
        return list(ModelFactory._CONFIGS.keys())

    @staticmethod
    def has_param_grid(model_name: str) -> bool:
        return model_name.lower() in _PARAM_GRIDS