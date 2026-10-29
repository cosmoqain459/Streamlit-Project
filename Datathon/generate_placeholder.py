"""
Generates working placeholder artifacts for the Streamlit decision-support dashboard
matching the exact schema of the Kaggle notebook.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def generate():
    np.random.seed(42)
    n_samples = 600

    # 1. Synthesize RFM distributions resembling Online Retail II
    recency = np.random.exponential(scale=100, size=n_samples) + 1
    frequency = np.random.geometric(p=0.15, size=n_samples)
    monetary = frequency * np.random.gamma(shape=3.0, scale=80.0, size=n_samples) + 10.0
    distinct_products = np.clip((frequency * np.random.uniform(5, 25, size=n_samples)).astype(int), 1, 300)
    tenure = recency + np.random.exponential(scale=150, size=n_samples)

    # Log transformations
    freq_log = np.log1p(frequency)
    monetary_log = np.log1p(monetary)
    distinct_log = np.log1p(distinct_products)

    # Countries
    countries = ['United Kingdom'] * 480 + ['Germany'] * 35 + ['France'] * 30 + ['Spain'] * 15 + ['Other'] * 40
    np.random.shuffle(countries)

    # Churn probability based on RFM rules (high recency -> high churn, high frequency -> low churn)
    churn_logits = (
        0.015 * recency 
        - 0.8 * freq_log 
        - 0.4 * monetary_log 
        - 0.3 * distinct_log 
        - 0.002 * tenure
    )
    churn_prob = 1 / (1 + np.exp(-churn_logits))
    churned = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        'Recency': recency.round(1),
        'Frequency_log': freq_log.round(4),
        'Monetary_log': monetary_log.round(4),
        'DistinctProducts_log': distinct_log.round(4),
        'Tenure': tenure.round(1),
        'CountryBucket': countries,
        'Churned': churned
    })

    numeric_features = ['Recency', 'Frequency_log', 'Monetary_log', 'DistinctProducts_log', 'Tenure']
    categorical_features = ['CountryBucket']

    X = df[numeric_features + categorical_features]
    y = df['Churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing
    scaler = StandardScaler().fit(X_train[numeric_features])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[categorical_features])

    X_train_proc = np.hstack([scaler.transform(X_train[numeric_features]), encoder.transform(X_train[categorical_features])])
    X_test_proc = np.hstack([scaler.transform(X_test[numeric_features]), encoder.transform(X_test[categorical_features])])

    # Model training
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_proc, y_train)

    # Holdout evaluation
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    test_preds = model.predict(X_test_proc)

    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds))
    rec = float(recall_score(y_test, test_preds))
    f1 = float(f1_score(y_test, test_preds))
    auc = float(roc_auc_score(y_test, test_probs))

    high_cutoff = float(np.percentile(test_probs, 80))
    medium_cutoff = float(np.percentile(test_probs, 50))

    cat_encoded = list(encoder.get_feature_names_out(categorical_features))
    all_names = numeric_features + cat_encoded
    importances = {k: round(float(v), 5) for k, v in zip(all_names, model.feature_importances_)}
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    artifacts_dir = os.path.join(os.path.dirname(__file__), 'app', 'artifacts')
    data_dir = os.path.join(os.path.dirname(__file__), 'app', 'data')
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save estimators
    joblib.dump(model, os.path.join(artifacts_dir, 'model.pkl'))
    joblib.dump(encoder, os.path.join(artifacts_dir, 'encoder.pkl'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.pkl'))

    # Save schema & metrics
    schema = {
        "target": "Churned",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "probability_class": 1,
        "model_type": "RandomForestClassifier",
        "probability_meaning": "Probability of Customer Churn (1 = Churned, 0 = Retained)"
    }
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "high_intent_cutoff": round(high_cutoff, 4),
        "medium_intent_cutoff": round(medium_cutoff, 4),
        "feature_importances": importances
    }

    with open(os.path.join(artifacts_dir, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)

    with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save sample test sessions
    sample_df = X_test.head(100).copy()
    sample_df.to_csv(os.path.join(data_dir, 'sample_sessions.csv'), index=False)

    print(f"Generated placeholder artifacts in {artifacts_dir}")
    print(f"Generated sample sessions in {data_dir}")

if __name__ == '__main__':
    generate()
