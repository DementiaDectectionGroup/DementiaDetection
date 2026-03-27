from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)[:, 1]
        else:
            raise AttributeError(f"{self.model.__class__.__name__} does not support predict_proba.")

    def measure(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        try:
            y_prob = self.predict_proba(X_test)
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except AttributeError:
            metrics['roc_auc'] = None

        return metrics

class ModelFactory:
    @staticmethod
    def get_model(model_name, **kwargs):
        """
        Returns an wrapped model based on the requested model name.
        
        Supported models:
        - 'logistic_regression'
        - 'svm'
        - 'random_forest'
        """
        model_name = model_name.lower()
        
        if model_name == 'logistic_regression':
            base_model = LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced',
                C=0.1,  # Adding slight regularization to prevent overfitting
                **kwargs
            )
            
        elif model_name == 'svm':
            base_model = SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42, 
                class_weight='balanced',
                C=1.0,
                gamma='scale',
                **kwargs
            )
            
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=42, 
                class_weight='balanced',
                n_estimators=200,     # Increase number of trees
                max_depth=10,         # Limit depth to prevent overfitting
                min_samples_split=5,  # Require more samples to split a node
                **kwargs
            )
            
        else:
            raise ValueError(f"Model '{model_name}' is not supported. "
                             f"Choose from: 'logistic_regression', 'svm', 'random_forest'.")

        return ModelWrapper(base_model)