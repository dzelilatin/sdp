import os
import pandas as pd
import numpy as np
from ml_analysis import prepare_features, train_models


def sample_df():
    # Minimal synthetic dataset
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'customerID': [f'id_{i}' for i in range(n)],
        'gender': rng.choice(['Male', 'Female'], size=n),
        'SeniorCitizen': rng.choice(['Yes', 'No'], size=n),
        'Partner': rng.choice(['Yes', 'No'], size=n),
        'Dependents': rng.choice(['Yes', 'No'], size=n),
        'tenure': rng.integers(0, 72, size=n),
        'InternetService': rng.choice(['DSL', 'Fiber optic', 'No'], size=n),
        'PaymentMethod': rng.choice(['Electronic check', 'Credit card', 'Bank transfer'], size=n),
        'MonthlyCharges': rng.normal(70, 30, size=n).clip(0),
        'TotalCharges': rng.normal(2000, 800, size=n).clip(0),
        'Churn': rng.choice(['Yes', 'No'], size=n, p=[0.25, 0.75]),
    })
    return df


def test_prepare_features_shapes():
    df = sample_df()
    X_train_df, X_test_sel, y_train_res, y_test, feat_names, preproc, selector, X_train_pre_df, y_train_pre = prepare_features(df)
    assert isinstance(X_train_df, pd.DataFrame)
    assert X_test_sel.shape[1] == len(feat_names)
    assert len(y_train_res) == len(X_train_df)
    assert len(y_test) > 0
    assert 'MonthlyChargesPerTenure' in preproc.transformers_[0][2] or 'MonthlyChargesPerTenure' in df.columns


def test_train_models_basic():
    df = sample_df()
    X_train_df, X_test_sel, y_train_res, y_test, feat_names, preproc, selector, X_train_pre_df, y_train_pre = prepare_features(df)
    results, models = train_models(X_train_df, X_test_sel, y_train_res, y_test, cv=2, randomized=True, best_only=True)
    assert isinstance(results, dict) and len(results) > 0
    any_acc = any('accuracy' in v and 0 <= v['accuracy'] <= 1 for v in results.values())
    assert any_acc
    assert isinstance(models, dict) and len(models) > 0 