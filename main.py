import numpy as np
from sklearn.preprocessing import StandardScaler
from dataprocess import load_or_extract_features, get_kfold_splits
from modelfactory import ModelFactory

def evaluate_models():
    train_audio_dir = "ADReSSo/train/audio"
    cache_file = "train_features_cache.npz"
    
    print("Loading data...")
    X, y = load_or_extract_features(train_audio_dir, cache_path=cache_file)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    models_to_test = ['logistic_regression', 'svm', 'random_forest']
    n_splits = 5
    
    results = {model: {'accuracy': [], 'f1_score': [], 'roc_auc': []} for model in models_to_test}
    
    print(f"\nStarting {n_splits}-Fold Cross Validation...\n")
    
    for fold, (train_idx, val_idx) in enumerate(get_kfold_splits(X, y, n_splits=n_splits)):
        print(f"--- Fold {fold + 1} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        for model_name in models_to_test:
            model = ModelFactory.get_model(model_name)
            model.fit(X_train, y_train)
            
            metrics = model.measure(X_val, y_val)
            
            results[model_name]['accuracy'].append(metrics['accuracy'])
            results[model_name]['f1_score'].append(metrics['f1_score'])
            if metrics['roc_auc'] is not None:
                results[model_name]['roc_auc'].append(metrics['roc_auc'])
                
            print(f"[{model_name.upper()}] Accuracy: {metrics['accuracy']:.4f}, "
                  f"F1-Score: {metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] else 'N/A'}")
        print()
        
    print("="*40)
    print("Final Average Results:")
    print("="*40)
    
    for model_name in models_to_test:
        avg_acc = np.mean(results[model_name]['accuracy'])
        avg_f1 = np.mean(results[model_name]['f1_score'])
        avg_auc = np.mean(results[model_name]['roc_auc']) if results[model_name]['roc_auc'] else 0.0
        
        print(f"Model: {model_name.upper()}")
        print(f"  Accuracy : {avg_acc:.4f}")
        print(f"  F1-Score : {avg_f1:.4f}")
        print(f"  ROC-AUC  : {avg_auc:.4f}\n")

if __name__ == "__main__":
    evaluate_models()