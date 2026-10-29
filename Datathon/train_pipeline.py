"""
Retail Customer Purchase & Churn Intelligence Engine
Problem Statement 2: AI/ML-Powered Predictive Insight Dashboard

End-to-end training and artifact generation pipeline for the multi-year Online Retail II dataset:
1. Memory-optimized chunked ingestion (safe for multi-GB files)
2. Leak-free RFM feature engineering with 75% cutoff date
3. Random Forest classifier training with balanced class weights
4. Unseen test set evaluation (ROC-AUC, F1, Accuracy, Precision, Recall)
5. Fixed calibrated cutoff derivation (80th and 50th percentile)
6. Export model artifacts, contracts, test batch, and 1-click ZIP
"""

import os
import glob
import json
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

def main():
    print("=" * 60)
    print("  RETAIL PURCHASE & CHURN PREDICTION PIPELINE")
    print("=" * 60)

    # ── Step 1: Discover Dataset & Set Optimized Dtypes ──
    possible_files = glob.glob('/kaggle/input/**/online_retail_II.csv', recursive=True) + \
                     glob.glob('/kaggle/input/**/online_retail*.csv', recursive=True)

    if possible_files:
        data_file = possible_files[0]
        print(f"[Step 1] Found dataset at: {data_file}")
    else:
        data_file = '/kaggle/input/datasets/mashlyn/online-retail-ii-uci/online_retail_II.csv'
        print(f"[Step 1] Using default path: {data_file}")

    dtype_spec = {
        'Invoice': 'str', 'InvoiceNo': 'str', 'StockCode': 'str', 'Description': 'str',
        'Quantity': 'float32', 'Price': 'float32', 'UnitPrice': 'float32',
        'Customer ID': 'float32', 'CustomerID': 'float32', 'Country': 'str'
    }

    # ── Step 2: Chunked Ingestion & Cleaning ──
    print("\n[Step 2] Ingesting transactions with RAM-safe streaming (150k rows/chunk)...")
    chunks = []
    total_raw_rows = 0
    chunk_size = 150_000
    rename_map = {'InvoiceNo': 'Invoice', 'UnitPrice': 'Price', 'CustomerID': 'Customer ID'}

    for chunk in pd.read_csv(data_file, encoding='ISO-8859-1', chunksize=chunk_size, dtype=dtype_spec, low_memory=False):
        total_raw_rows += len(chunk)
        chunk = chunk.rename(columns={k: v for k, v in rename_map.items() if k in chunk.columns})
        chunk = chunk.dropna(subset=['Customer ID'])
        chunk = chunk[~chunk['Invoice'].astype(str).str.startswith('C')]
        chunk = chunk[(chunk['Quantity'] > 0) & (chunk['Price'] > 0)]
        chunk['Customer ID'] = chunk['Customer ID'].astype(np.int32)
        chunk['InvoiceDate'] = pd.to_datetime(chunk['InvoiceDate'])
        chunk['LineTotal'] = (chunk['Quantity'] * chunk['Price']).astype(np.float32)
        clean_cols = ['Customer ID', 'Invoice', 'StockCode', 'InvoiceDate', 'LineTotal', 'Country']
        chunks.append(chunk[clean_cols])

    df_clean = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"  Raw rows: {total_raw_rows:,} -> Cleaned rows: {df_clean.shape[0]:,}")
    print(f"  Unique customers: {df_clean['Customer ID'].nunique():,}")

    # ── Step 3: Time-Based Split (Leakage Prevention) ──
    full_start = df_clean['InvoiceDate'].min()
    full_end = df_clean['InvoiceDate'].max()
    total_days = (full_end - full_start).days
    cutoff_date = full_start + pd.Timedelta(days=int(total_days * 0.75))
    print(f"\n[Step 3] Timeline: {full_start} to {full_end} ({total_days} days)")
    print(f"  75% Cutoff Date: {cutoff_date}")

    feature_window = df_clean[df_clean['InvoiceDate'] <= cutoff_date]
    label_window = df_clean[df_clean['InvoiceDate'] > cutoff_date]

    # ── Step 4: RFM Feature Engineering ──
    print("\n[Step 4] Engineering RFM customer profiles...")
    snapshot_date = cutoff_date + pd.Timedelta(days=1)
    rfm = feature_window.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('LineTotal', 'sum'),
        DistinctProducts=('StockCode', 'nunique'),
        Tenure=('InvoiceDate', lambda x: (snapshot_date - x.min()).days),
        TopCountry=('Country', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    ).reset_index()
    print(f"  Engineered profiles for {rfm.shape[0]:,} customers.")

    # ── Step 5: Ground-Truth Churn Label ──
    returning_customers = set(label_window['Customer ID'].unique())
    rfm['Churned'] = (~rfm['Customer ID'].isin(returning_customers)).astype(int)
    print(f"\n[Step 5] Overall churn rate: {rfm['Churned'].mean():.2%}")

    # ── Step 6: Log Transforms & Country Bucketing ──
    print("\n[Step 6] Applying log1p transformations and country bucketing...")
    for col in ['Monetary', 'DistinctProducts', 'Frequency']:
        rfm[f'{col}_log'] = np.log1p(np.maximum(0, rfm[col]))

    country_counts = rfm['TopCountry'].value_counts()
    common_countries = country_counts[country_counts >= 15].index.tolist()
    rfm['CountryBucket'] = rfm['TopCountry'].apply(lambda x: x if x in common_countries else 'Other')

    # ── Step 7: Stratified Train/Test Split ──
    numeric_features = ['Recency', 'Frequency_log', 'Monetary_log', 'DistinctProducts_log', 'Tenure']
    categorical_features = ['CountryBucket']
    X = rfm[numeric_features + categorical_features].copy()
    y = rfm['Churned'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[Step 7] Split sizes: Train={X_train.shape[0]:,} | Test={X_test.shape[0]:,}")

    scaler = StandardScaler().fit(X_train[numeric_features])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[categorical_features])

    X_train_proc = np.hstack([scaler.transform(X_train[numeric_features]), encoder.transform(X_train[categorical_features])])
    X_test_proc = np.hstack([scaler.transform(X_test[numeric_features]), encoder.transform(X_test[categorical_features])])

    # ── Step 8: Train Balanced Random Forest ──
    print("\n[Step 8] Training balanced Random Forest (150 trees)...")
    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_proc, y_train)

    # ── Step 9: Evaluate on Holdout Set ──
    print("\n[Step 9] Evaluating holdout test set...")
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    test_preds = model.predict(X_test_proc)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_probs)

    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Accuracy:  {acc:.4f}")

    # ── Step 10: Calibrated Intent Cutoffs ──
    high_cutoff = float(np.percentile(test_probs, 80))
    medium_cutoff = float(np.percentile(test_probs, 50))
    print(f"\n[Step 10] Cutoffs: High >= {high_cutoff:.4f} | Medium >= {medium_cutoff:.4f}")

    # ── Step 11: Feature Importances ──
    cat_encoded_names = list(encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_encoded_names
    feature_importances_dict = {
        feat: round(float(imp), 5) for feat, imp in zip(all_feature_names, model.feature_importances_)
    }
    feature_importances_sorted = dict(sorted(feature_importances_dict.items(), key=lambda item: item[1], reverse=True))

    # ── Step 12: Contract Payloads ──
    schema = {
        "target": "Churned",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "probability_class": 1,
        "model_type": "RandomForestClassifier",
        "probability_meaning": "Probability of Customer Churn (1 = Churned, 0 = Retained)"
    }
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "high_intent_cutoff": round(float(high_cutoff), 4),
        "medium_intent_cutoff": round(float(medium_cutoff), 4),
        "feature_importances": feature_importances_sorted
    }

    # ── Step 13: Export Artifacts ──
    artifacts_dir = '/kaggle/working/artifacts' if os.path.exists('/kaggle') else 'app/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifacts_dir, 'model.pkl'))
    joblib.dump(encoder, os.path.join(artifacts_dir, 'encoder.pkl'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.pkl'))
    with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(artifacts_dir, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"\n[Step 13] Saved 5 artifacts to {artifacts_dir}")

    # ── Step 14: Export Sample Inference Batch ──
    data_dir = '/kaggle/working/data' if os.path.exists('/kaggle') else 'app/data'
    os.makedirs(data_dir, exist_ok=True)
    sample_file = os.path.join(data_dir, 'sample_sessions.csv')
    X_test.head(100).to_csv(sample_file, index=False)
    print(f"[Step 14] Saved 100 test rows to {sample_file}")

    # ── Step 15: Create ZIP Package ──
    if os.path.exists('/kaggle'):
        zip_filename = '/kaggle/working/retail_intelligence_artifacts'
        shutil.make_archive(zip_filename, 'zip', '/kaggle/working/artifacts')
        print(f"[Step 15] Download package ready at {zip_filename}.zip")

    print("\nPIPELINE EXECUTION COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
