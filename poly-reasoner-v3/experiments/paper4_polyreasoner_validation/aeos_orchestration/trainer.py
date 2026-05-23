"""
AITL V2 — Model-Agnostic Trainer
Executes ANY Python code the agent generates (sklearn, PyTorch, raw numpy, etc.)
Measures results externally — agent can't cheat the metric.
"""
import traceback
import signal
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import ConvergenceWarning

# Suppress noisy sklearn warnings that pollute terminal output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import multiprocessing

def _run_in_process(code_str, X_train, y_train, X_val, y_val, return_dict):
    """Executes the agent code inside a separate process."""
    import traceback
    import numpy as np
    
    exec_namespace = {
        '__builtins__': __builtins__,
        'np': np,
        'numpy': np,
    }
    
    # Inject libraries that are available
    try:
        import sklearn
        exec_namespace['sklearn'] = sklearn
        from sklearn import ensemble, linear_model, svm, tree, neighbors
        from sklearn import preprocessing, pipeline, model_selection
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.model_selection import cross_val_score
        
        exec_namespace.update({
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier,
            'SVC': SVC,
            'LinearSVC': LinearSVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'MLPClassifier': MLPClassifier,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'Pipeline': Pipeline,
            'TfidfVectorizer': TfidfVectorizer,
            'CountVectorizer': CountVectorizer,
            'PCA': PCA,
            'TruncatedSVD': TruncatedSVD,
            'VotingClassifier': VotingClassifier,
            'StackingClassifier': StackingClassifier,
            'BaggingClassifier': BaggingClassifier,
            'cross_val_score': cross_val_score,
        })
    except ImportError:
        pass
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        exec_namespace.update({
            'torch': torch, 'nn': nn, 'F': F, 'optim': optim,
        })
    except ImportError:
        pass

    # HuggingFace transformers (for BERT-specialist coder experiments)
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
        from transformers import DistilBertModel, DistilBertTokenizer
        from transformers import BertModel, BertTokenizer
        exec_namespace.update({
            'transformers': transformers,
            'AutoModel': AutoModel,
            'AutoTokenizer': AutoTokenizer,
            'AutoModelForSequenceClassification': AutoModelForSequenceClassification,
            'DistilBertModel': DistilBertModel,
            'DistilBertTokenizer': DistilBertTokenizer,
            'BertModel': BertModel,
            'BertTokenizer': BertTokenizer,
        })
    except ImportError:
        pass

    try:
        exec(code_str, exec_namespace)
        solve_fn = exec_namespace.get('solve')
        if not solve_fn:
            return_dict['error'] = "Structure Error: Must define 'def solve(X_train, y_train, X_val, y_val)' that returns predictions."
            return
            
        predictions = solve_fn(
            X_train.copy(), y_train.copy(), 
            X_val.copy(), y_val.copy()
        )
        return_dict['predictions'] = predictions
    except Exception as e:
        return_dict['error'] = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"


def execute_agent_code(code_str, X_train, y_train, X_val, y_val, n_classes, timeout=300):
    """
    Execute agent-generated code and measure results.
    
    The agent must define:
        def solve(X_train, y_train, X_val, y_val):
            # ... any code ...
            return predictions  # numpy array, shape (n_val,)
    
    Returns:
        (results_dict, None) on success
        (None, error_string) on failure
    """
    try:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        p = multiprocessing.Process(
            target=_run_in_process,
            args=(code_str, X_train, y_train, X_val, y_val, return_dict)
        )
        p.start()
        p.join(timeout=timeout)
        
        if p.is_alive():
            p.terminate()
            p.join()
            return None, f"Timeout Error: Code execution exceeded {timeout} seconds. Process terminated."
        
        if 'error' in return_dict:
            return None, return_dict['error']
            
        if 'predictions' not in return_dict:
            return None, "Error: process terminated unexpectedly without returning predictions."
            
        predictions = return_dict['predictions']
        
        # Validate predictions
        if predictions is None:
            return None, "Error: solve() returned None. Must return predictions array."
        
        predictions = np.array(predictions).flatten()
        
        if len(predictions) != len(y_val):
            return None, f"Shape Error: Expected {len(y_val)} predictions, got {len(predictions)}."
        
        # Ensure predictions are integer class labels
        predictions = predictions.astype(int)
        
        # Validate prediction range
        unique_preds = np.unique(predictions)
        if np.any(unique_preds < 0) or np.any(unique_preds >= n_classes):
            return None, f"Range Error: Predictions must be in [0, {n_classes-1}], got range [{unique_preds.min()}, {unique_preds.max()}]."
        
        # --- External measurement (agent can't influence this) ---
        acc = accuracy_score(y_val, predictions)
        
        # Compute log-loss from predictions (convert to one-hot probabilities)
        lb = LabelBinarizer()
        lb.fit(range(n_classes))
        pred_onehot = lb.transform(predictions)
        # Add small epsilon to avoid log(0)
        pred_proba = pred_onehot * 0.95 + (1 - pred_onehot) * (0.05 / (n_classes - 1))
        loss = log_loss(y_val, pred_proba)
        
        return {
            "val_accuracy": round(acc, 4),
            "val_loss": round(loss, 4),
            "n_unique_predictions": len(unique_preds),
            "prediction_distribution": np.bincount(predictions, minlength=n_classes).tolist(),
        }, None
        
    except Exception as e:
        return None, f"Execution Error: {str(e)}\n{traceback.format_exc()}"
