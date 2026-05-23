"""
AEOS — Multi-Dataset Loader
Provides 3 datasets designed to test different LLM capabilities:
  1. tabular  → Covtype (54 features, 7 classes, needs scaling)
  2. text     → 20 Newsgroups (raw strings, needs TF-IDF vectorization)
  3. vision   → MNIST digits (784 pixel features, needs PCA or CNN)

Each dataset returns:
  X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint
  
  dataset_hint: a string injected into the LLM's system prompt so it knows
                what kind of data it's working with (without giving away the answer).
"""
import numpy as np
from sklearn.model_selection import train_test_split


# ─── Dataset hint strings (injected into the agent's system prompt) ─────────
HINTS = {
    "tabular": (
        "This is a TABULAR dataset. X contains numeric features (floats/ints). "
        "The features may have different scales. Consider preprocessing like "
        "StandardScaler or normalization. Tree-based models, SVMs, and neural "
        "networks are all viable approaches."
    ),
    "tabular2": (
        "This is a generic TABULAR classification dataset. "
        "X contains numeric features (all floats). You have no information about the domain. "
        "You must figure out the best preprocessing, feature engineering, and model architecture entirely on your own.\n\n"
        "CRITICAL RULES (violations = instant failure):\n"
        "1. Do NOT call train_test_split on X_train/y_train inside solve() — the split is already done for you.\n"
        "2. For multiclass, use model.predict(X_val) NOT predict_proba(X_val)[:, 1] — that is binary-only.\n"
        "3. Do NOT import imblearn, tensorflow, or keras — they are NOT installed.\n"
        "4. Do NOT call solve() at module level outside the function definition.\n"
        "5. Do NOT use GridSearchCV — it will cause a timeout. Use fixed hyperparameters.\n"
        "6. Do NOT use max_features='auto' in RandomForestClassifier — deprecated. Use 'sqrt' instead."
    ),
    "text": (
        "This is a TEXT dataset. X_train and X_val are 1D arrays of raw text strings "
        "(NOT numeric arrays). You CANNOT pass them directly to model.fit(). "
        "You MUST convert text to numeric features first using something like "
        "TfidfVectorizer, CountVectorizer, or similar NLP preprocessing. "
        "Example: from sklearn.feature_extraction.text import TfidfVectorizer"
    ),
    "vision": (
        "This is a VISION dataset. Each sample is a flattened image (784 pixel values, "
        "ranging 0-255). The original image shape is 28x28 pixels. "
        "You can treat it as a flat 784-feature vector (for sklearn models), "
        "or reshape to (28, 28) or (1, 28, 28) for convolutional neural networks. "
        "Consider: PCA for dimensionality reduction, pixel normalization (/255), "
        "or building a CNN with PyTorch."
    ),
}


def get_data(dataset="tabular", n_samples=10000, seed=42):
    """
    Load one of 3 benchmark datasets.
    
    Args:
        dataset: 'tabular', 'text', or 'vision'
        n_samples: subsample size (for speed)
        seed: random seed for reproducibility
        
    Returns:
        X_train, y_train, X_val, y_val, n_features, n_classes, dataset_hint
    """
    if dataset == "tabular":
        return _load_tabular(n_samples, seed)
    elif dataset == "tabular2":
        return _load_tabular2(n_samples, seed)
    elif dataset == "text":
        return _load_text(n_samples, seed)
    elif dataset == "vision":
        return _load_vision(n_samples, seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: tabular, tabular2, text, vision")


def _load_tabular(n_samples, seed):
    """Covtype: 54 features, 7 classes. Tests scaling and non-linear models."""
    from sklearn.datasets import fetch_covtype
    
    print("  [Data] Loading Cover Type (tabular) dataset...")
    data = fetch_covtype()
    X_full, y_full = data.data, data.target
    
    # Covtype classes are 1-7, remap to 0-6
    y_full = y_full - 1
    
    # Stratified subsample
    if n_samples < len(X_full):
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full, train_size=n_samples,
            random_state=seed, stratify=y_full
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
    )
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    _print_stats("tabular (Covtype)", X_train, X_val, y_train, n_features, n_classes)
    return X_train, y_train.astype(int), X_val, y_val.astype(int), n_features, n_classes, HINTS["tabular"]


def _load_tabular2(n_samples, seed):
    """Dry Bean: 16 morphological features, 7 classes. Fresh dataset for EXP2.
    
    Source: UCI ML Repository — Dry Bean Dataset (Koklu & Ozkan, 2020).
    Downloaded via ucimlrepo or sklearn fetch. The agent sees only raw
    numeric features with no semantic labels (full blind-dataset protocol).
    """
    print("  [Data] Loading Dry Bean (tabular2) dataset...")
    
    try:
        # Primary: ucimlrepo (pip install ucimlrepo)
        from ucimlrepo import fetch_ucirepo
        dry_bean = fetch_ucirepo(id=602)
        X_full = dry_bean.data.features.values.astype(np.float64)
        # Encode string class labels to integers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_full = le.fit_transform(dry_bean.data.targets.values.ravel())
        print(f"  [Data] Classes: {list(le.classes_)} → mapped to 0-{len(le.classes_)-1}")
    except Exception as e:
        print(f"  [Data] ucimlrepo unavailable ({e}). Falling back to OpenML...")
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(name='Dry_Bean_Dataset', version=1, as_frame=True, parser='auto')
        X_full = ds.data.values.astype(np.float64)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_full = le.fit_transform(ds.target.values)
    
    # Stratified subsample
    if n_samples < len(X_full):
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full, train_size=n_samples,
            random_state=seed, stratify=y_full
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
    )
    
    n_features = X_train.shape[1]   # 16
    n_classes = len(np.unique(y_train))  # 7
    
    _print_stats("tabular2 (Dry Bean)", X_train, X_val, y_train, n_features, n_classes)
    return X_train, y_train.astype(int), X_val, y_val.astype(int), n_features, n_classes, HINTS["tabular2"]


def _load_text(n_samples, seed):
    """20 Newsgroups: raw text strings, 20 classes. Tests NLP pipeline awareness."""
    from sklearn.datasets import fetch_20newsgroups
    
    print("  [Data] Loading 20 Newsgroups (text) dataset...")
    
    # Use a subset of 6 categories to keep it manageable but still challenging
    categories = [
        'comp.graphics', 'sci.med', 'rec.sport.baseball',
        'talk.politics.misc', 'sci.space', 'rec.autos'
    ]
    
    train_data = fetch_20newsgroups(subset='train', categories=categories,
                                     remove=('headers', 'footers', 'quotes'),
                                     random_state=seed)
    test_data = fetch_20newsgroups(subset='test', categories=categories,
                                    remove=('headers', 'footers', 'quotes'),
                                    random_state=seed)
    
    X_train = np.array(train_data.data)
    y_train = train_data.target
    X_val = np.array(test_data.data)
    y_val = test_data.target
    
    # Subsample if needed
    if n_samples < len(X_train):
        idx = np.random.RandomState(seed).choice(len(X_train), n_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    n_features = 0  # Text — no fixed feature count
    n_classes = len(np.unique(y_train))
    
    print(f"  [Data] Loaded: {len(X_train)} train, {len(X_val)} val")
    print(f"  [Data] Classes: {n_classes} (text categories)")
    print(f"  [Data] Sample text (first 80 chars): {X_train[0][:80]}...")
    print("  [Data] WARNING: X is an array of RAW STRINGS - not numeric!")
    
    return X_train, y_train.astype(int), X_val, y_val.astype(int), n_features, n_classes, HINTS["text"]


def _load_vision(n_samples, seed):
    """MNIST digits: 784 pixel features, 10 classes. Tests dimensionality awareness."""
    from sklearn.datasets import fetch_openml
    
    print("  [Data] Loading MNIST (vision) dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_full, y_full = mnist.data.astype(np.float32), mnist.target.astype(int)
    
    # Stratified subsample
    if n_samples < len(X_full):
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full, train_size=n_samples,
            random_state=seed, stratify=y_full
        )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
    )
    
    n_features = X_train.shape[1]  # 784
    n_classes = len(np.unique(y_train))  # 10
    
    _print_stats("vision (MNIST)", X_train, X_val, y_train, n_features, n_classes)
    print(f"  [Data] Pixel value range: [{X_train.min():.0f}, {X_train.max():.0f}]")
    
    return X_train, y_train.astype(int), X_val, y_val.astype(int), n_features, n_classes, HINTS["vision"]


def _print_stats(name, X_train, X_val, y_train, n_features, n_classes):
    """Common stats printer."""
    print(f"  [Data] Loaded: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"  [Data] Features: {n_features}, Classes: {n_classes}")
    print(f"  [Data] Class distribution: {np.bincount(y_train.astype(int))}")
