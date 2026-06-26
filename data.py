import torch
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset, Subset
import torchvision
from torchvision import datasets, transforms as T
import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold
import os

#  CoverType data set
def covertype_dataloaders(
    batch_size: int = 1024,
    test_size: float = 0.15,
    val_size: float = 0.15,   # fraction of the *remaining* train after test split
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Gorishniy-style preprocessing for the MLP baseline:
      - dense float features
      - StandardScaler fit on train, applied to val/test
      - stratified splits
      - returns PyTorch DataLoaders

    Returns:
      train_loader, val_loader, test_loader, d_in, n_classes
    """
   
    data = fetch_covtype()
    X = data.data.astype(np.float32)                  # shape (N, 54)
    y = (data.target.astype(np.int64) - 1)            # classes: 0..6

    d_in = X.shape[1]
    n_classes = int(np.max(y) + 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t   = torch.from_numpy(X_val)
    y_val_t   = torch.from_numpy(y_val)
    X_test_t  = torch.from_numpy(X_test)
    y_test_t  = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t, y_val_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, drop_last=False
    )

    return train_loader, val_loader, test_loader, d_in, n_classes


def covertype_dataloaders_kfold(
    batch_size: int = 1024,
    test_size: float = 0.15,
    k: int = 5,
    seed: int = 42,
):
    """
    Correct CV protocol for tabular data:
      - Fixed held-out test split
      - K-fold on remaining train split
      - Per-fold StandardScaler fit on fold-train only
      - Test is transformed per fold using that fold scaler (=> per-fold test_loader)

    Returns:
      folds, d_in, n_classes
      folds = [(train_loader, val_loader, test_loader_fold), ...] length k
    """

    data = fetch_covtype()
    X = data.data.astype(np.float32)          # (N, 54)
    y = (data.target.astype(np.int64) - 1)    # classes 0..6

    d_in = X.shape[1]
    n_classes = int(np.max(y) + 1)

    # ---- Fixed test split ----
    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y 
    )
    

    # ---- Fold splitter ----
    
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    split_iter = splitter.split(X_trainfull, y_trainfull)
    use_pin = torch.cuda.is_available()
    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(split_iter):
        X_tr = X_trainfull[train_idx]
        y_tr = y_trainfull[train_idx]
        X_va = X_trainfull[val_idx]
        y_va = y_trainfull[val_idx]

        # ---- Per-fold scaler ----
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)
        X_te_s = scaler.transform(X_test).astype(np.float32)

        # ---- TensorDatasets ----
        train_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
        val_ds   = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va))
        test_ds  = TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(y_test))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=use_pin, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=use_pin, drop_last=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=use_pin, drop_last=False
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds, d_in, n_classes


# ─────────────────────────────────────────────────────────────────────────────
#  Adult data set (ordinal encoding)
# ─────────────────────────────────────────────────────────────────────────────

# Feature split used by the ordinal Adult pipeline: 6 numeric + 8 categorical
# columns -> 14-dimensional input (categoricals are integer-encoded, not one-hot).
ADULT_NUM_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]
ADULT_CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def _load_adult_raw():
    """
    Fetch the Adult dataset from OpenML, drop rows with missing values, and
    return raw arrays: X_num (N, 6) float, X_cat (N, 8) str, y (N,) int
    (1 = income >50K). Categoricals are kept as strings for the OrdinalEncoder.
    """
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()

    target_col = "class"
    df[target_col] = (df[target_col].astype(str).str.strip() == ">50K").astype(int)
    df = df.dropna(subset=ADULT_CAT_FEATURES + ADULT_NUM_FEATURES + [target_col])

    X_num = df[ADULT_NUM_FEATURES].astype(float).values   # (N, 6) float
    X_cat = df[ADULT_CAT_FEATURES].astype(str).values     # (N, 8) str
    y     = df[target_col].values.astype(np.int64)        # (N,)
    return X_num, X_cat, y


def _build_ordinal_preprocessor(X_num_tr, X_cat_tr, global_categories, seed):
    """
    Fit the ordinal preprocessor on a single split (no leakage):
      - QuantileTransformer (normal output) on the numeric columns,
      - OrdinalEncoder on the categoricals using a FIXED global vocabulary so the
        encoding is consistent across folds; unseen categories map to -1.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", QuantileTransformer(
            n_quantiles=min(1000, len(X_num_tr)),
            output_distribution="normal",
            random_state=seed,
            subsample=int(1e9),
        ), list(range(len(ADULT_NUM_FEATURES)))),
        ("cat", OrdinalEncoder(
            categories=global_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        ), list(range(len(ADULT_NUM_FEATURES),
                      len(ADULT_NUM_FEATURES) + len(ADULT_CAT_FEATURES)))),
    ])
    preprocessor.fit(np.hstack([X_num_tr, X_cat_tr]))
    return preprocessor


def _apply_ordinal_preprocessor(preprocessor, X_num, X_cat):
    """Transform (X_num, X_cat) into a float32 array of shape (N, 14)."""
    return preprocessor.transform(np.hstack([X_num, X_cat])).astype(np.float32)


def adult_ordinal_dataloaders_kfold(
    teacher_ckpt: str,
    batch_size: int = 256,
    eval_bs: int = 8192,
    k: int = 5,
    seed: int = 6,
):
    """
    K-fold loaders for the ORDINAL Adult dataset, reproducing the exact splits
    used to train the published students.

    The fixed held-out test split, the development indices, the global category
    vocabulary, and the test-set preprocessor are read from the teacher
    checkpoint (`teacher_ckpt`) so that fold construction is identical to
    training time. Per fold: a fresh preprocessor is fit on the fold's train
    split (numerics only — no leakage); the fixed test set is transformed with
    the teacher's preprocessor for consistent evaluation.

    Mirrors `covertype_dataloaders_kfold`'s return contract.

    Returns:
        folds, d_in, n_classes
        folds = [(train_loader, val_loader, test_loader), ...] length k
    """
    # The checkpoint stores sklearn objects (preprocessor), so weights_only=False.
    ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    idx_dev           = ckpt["idx_dev"]
    idx_test          = ckpt["idx_test"]
    global_categories = ckpt["global_categories"]
    teacher_prep      = ckpt["preprocessor"]   # fixed test-set preprocessor

    X_num, X_cat, y = _load_adult_raw()
    d_in = len(ADULT_NUM_FEATURES) + len(ADULT_CAT_FEATURES)   # 14
    n_classes = int(np.max(y) + 1)                             # 2

    # Test split is fixed across folds; transform it once with the teacher prep.
    X_test = _apply_ordinal_preprocessor(teacher_prep, X_num[idx_test], X_cat[idx_test])
    y_test = y[idx_test]
    use_pin = torch.cuda.is_available()

    def _loader(X, yy, bs, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(yy))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=0, pin_memory=use_pin, drop_last=False)

    test_loader = _loader(X_test, y_test, eval_bs, shuffle=False)

    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_rel, val_rel in splitter.split(idx_dev, y[idx_dev]):
        abs_tr, abs_val = idx_dev[train_rel], idx_dev[val_rel]

        # Fit fold preprocessor on the fold's train split only.
        fold_prep = _build_ordinal_preprocessor(
            X_num[abs_tr], X_cat[abs_tr], global_categories, seed
        )
        X_tr  = _apply_ordinal_preprocessor(fold_prep, X_num[abs_tr],  X_cat[abs_tr])
        X_val = _apply_ordinal_preprocessor(fold_prep, X_num[abs_val], X_cat[abs_val])

        train_loader = _loader(X_tr,  y[abs_tr],  batch_size, shuffle=True)
        val_loader   = _loader(X_val, y[abs_val], eval_bs,    shuffle=False)
        folds.append((train_loader, val_loader, test_loader))

    return folds, d_in, n_classes


# ─────────────────────────────────────────────────────────────────────────────
#  HIGGS-Small data set
# ─────────────────────────────────────────────────────────────────────────────

def _load_higgs_small():
    """
    Fetch HIGGS-Small from OpenML (dataset 23512: 98k rows, 28 numeric features,
    binary target), drop rows with NaN features, and return X (float32) and
    integer-encoded y.
    """
    import openml  # imported lazily so the rest of data.py has no hard dependency
    from sklearn.preprocessing import LabelEncoder

    dataset = openml.datasets.get_dataset(23512)
    X, y, _, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    y = LabelEncoder().fit_transform(y).astype(np.int64)
    X = X.astype(np.float32)

    valid = ~np.isnan(X).any(axis=1)
    return X[valid], y[valid]


def higgs_small_dataloaders_kfold(
    n_splits: int = 5,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    K-fold loaders for HIGGS-Small, reproducing the splits used to train the
    published students. Mirrors `covertype_dataloaders_kfold`'s return contract.

    A fixed 15% test set is held out before the K-fold split; the test set is
    scaled with a QuantileTransformer fit on all dev data (consistent with the
    teacher), while each fold fits its own scaler on the fold's train split only
    (no leakage). QuantileTransformer uses `output_distribution='normal'`.

    Returns:
        folds, d_in, n_classes
        folds = [(train_loader, val_loader, test_loader), ...] length n_splits
                (test_loader is the same held-out set across folds)
    """
    X, y = _load_higgs_small()
    use_pin = torch.cuda.is_available()

    # Fixed held-out test split.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    # Test set is scaled with a scaler fit on all dev data (teacher-consistent).
    teacher_scaler = QuantileTransformer(
        output_distribution="normal", n_quantiles=min(len(X_dev), 1000), random_state=seed
    )
    teacher_scaler.fit(X_dev)
    X_test_sc = teacher_scaler.transform(X_test).astype(np.float32)

    def _loader(X, yy, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(yy))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=use_pin)

    test_loader = _loader(X_test_sc, y_test, shuffle=False)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in splitter.split(X_dev, y_dev):
        X_tr, y_tr = X_dev[train_idx], y_dev[train_idx]
        X_va, y_va = X_dev[val_idx], y_dev[val_idx]

        # Fold scaler fit on the fold's train split only.
        scaler = QuantileTransformer(
            output_distribution="normal", n_quantiles=min(len(X_tr), 1000), random_state=seed
        )
        X_tr_sc = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_sc = scaler.transform(X_va).astype(np.float32)

        train_loader = _loader(X_tr_sc, y_tr, shuffle=True)
        val_loader   = _loader(X_va_sc, y_va, shuffle=False)
        folds.append((train_loader, val_loader, test_loader))

    d_in = X.shape[1]                 # 28
    n_classes = int(np.max(y) + 1)    # 2
    return folds, d_in, n_classes

