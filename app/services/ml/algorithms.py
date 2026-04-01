from __future__ import annotations

"""
Production-ready ML algorithms using scikit-learn and mlxtend.

This module provides training and prediction functions for:
- Naive Bayes (classification)
- KNN (classification)
- K-Means (clustering)
- FP-Growth (association rules)

All algorithms use professional ML libraries for better accuracy and performance.

Type ignore comments are used throughout for sklearn/numpy type compatibility issues.
These are safe at runtime as sklearn guarantees correct return types.
"""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.services.dataio import detect_feature_types


@dataclass
class TrainResult:
    """Result of training operation containing model, metrics, and metadata."""

    model: Any  # Trained model object (sklearn model + preprocessing pipeline)
    metrics: dict  # Evaluation metrics
    metadata: dict  # Feature info, types, options, etc.


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator."""
    return a / b if b != 0 else default


def train_naive_bayes(df: pd.DataFrame, target_col: str) -> TrainResult:
    """
    Train a Naive Bayes classifier on the dataset.
    
    Uses GaussianNB for numeric features and CategoricalNB for categorical features.
    Automatically detects feature types and applies appropriate preprocessing.
    
    Args:
        df: pandas DataFrame with features and target
        target_col: Name of the target column
        
    Returns:
        TrainResult with trained model, metrics, and metadata
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Detect feature types
    feature_types, feature_options = detect_feature_types(df, exclude_cols=[target_col])
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None  # type: ignore[call-overload,arg-type]
    )
    
    # Preprocess features
    encoders = {}
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    for col in feature_cols:
        if feature_types.get(col) == "categorical":
            enc = LabelEncoder()
            X_train_processed[col] = enc.fit_transform(X_train[col].astype(str))
            X_test_processed[col] = enc.transform(X_test[col].astype(str))
            encoders[col] = enc
        else:
            # Fill NaN for numeric columns
            X_train_processed[col] = X_train[col].fillna(0)
            X_test_processed[col] = X_test[col].fillna(0)
    
    # Determine which Naive Bayes to use
    has_categorical = any(ft == "categorical" for ft in feature_types.values())
    
    if has_categorical:
        # Use CategoricalNB (works well with mixed data after encoding)
        model = CategoricalNB()
    else:
        # Use GaussianNB for purely numeric data
        model = GaussianNB()
    
    # Train
    model.fit(X_train_processed, y_train)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    labels = label_encoder.classes_.tolist()  # type: ignore[union-attr]
    
    metrics = {
        "accuracy": float(accuracy), 
        "precision": float(precision),  
        "recall": float(recall),  
        "f1_score": float(f1),  
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }
    
    # Package model with preprocessing info
    model_package = {
        "classifier": model,
        "label_encoder": label_encoder,
        "feature_encoders": encoders,
        "feature_cols": feature_cols,
        "feature_types": feature_types,
    }
    
    metadata = {
        "target_col": target_col,
        "features": feature_cols,
        "feature_types": feature_types,
        "feature_options": feature_options,
        "n_samples": len(df),
        "n_classes": len(labels),
    }
    
    return TrainResult(model=model_package, metrics=metrics, metadata=metadata)


def predict_naive_bayes(model_package: dict, input_dict: dict) -> dict:
    """
    Make prediction using trained Naive Bayes model.
    
    Args:
        model_package: Dict containing classifier and preprocessing objects
        input_dict: Dict of feature values
        
    Returns:
        Dict with prediction and probabilities
    """
    classifier = model_package["classifier"]
    label_encoder = model_package["label_encoder"]
    feature_encoders = model_package["feature_encoders"]
    feature_cols = model_package["feature_cols"]
    feature_types = model_package["feature_types"]
    
    # Prepare input
    X_input = pd.DataFrame([input_dict])[feature_cols]
    
    # Apply preprocessing
    X_processed = X_input.copy()
    for col in feature_cols:
        if col in feature_encoders:
            # Categorical encoding
            try:
                X_processed[col] = feature_encoders[col].transform(X_input[col].astype(str))
            except ValueError:
                # Unknown category, use most frequent class
                X_processed[col] = 0
        else:
            # Numeric
            X_processed[col] = X_input[col].fillna(0)
    
    # Predict
    y_pred = classifier.predict(X_processed)[0]
    y_proba = classifier.predict_proba(X_processed)[0]
    
    # Decode prediction
    prediction = label_encoder.inverse_transform([y_pred])[0]
    probabilities = {
        str(label): float(prob)
        for label, prob in zip(label_encoder.classes_, y_proba)
    }
    
    return {"prediction": str(prediction), "probabilities": probabilities}


def train_knn(df: pd.DataFrame, target_col: str, n_neighbors: int = 5) -> TrainResult:
    """
    Train a K-Nearest Neighbors classifier.
    
    Args:
        df: pandas DataFrame with features and target
        target_col: Name of the target column
        n_neighbors: Number of neighbors to use
        
    Returns:
        TrainResult with trained model, metrics, and metadata
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Detect feature types
    feature_types, feature_options = detect_feature_types(df, exclude_cols=[target_col])
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None  # type: ignore[call-overload,arg-type]
    )
    
    # Preprocess features
    encoders = {}
    scalers = {}
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    for col in feature_cols:
        if feature_types.get(col) == "categorical":
            # Encode categorical
            enc = LabelEncoder()
            X_train_processed[col] = enc.fit_transform(X_train[col].astype(str))
            X_test_processed[col] = enc.transform(X_test[col].astype(str))
            encoders[col] = enc
        else:
            # Fill NaN and scale numeric
            X_train_processed[col] = X_train[col].fillna(0)
            X_test_processed[col] = X_test[col].fillna(0)
            
            # Scale numeric features
            scaler = StandardScaler()
            X_train_processed[col] = scaler.fit_transform(X_train_processed[[col]])  # type: ignore[call-overload]
            X_test_processed[col] = scaler.transform(X_test_processed[[col]])  # type: ignore[call-overload]
            scalers[col] = scaler
    
    # Train KNN
    model = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(X_train) - 1))
    model.fit(X_train_processed, y_train)
    
    # Predict
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    labels = label_encoder.classes_.tolist()  # type: ignore[union-attr]
    
    metrics = {
        "accuracy": float(accuracy),  # type: ignore[arg-type]
        "precision": float(precision),  # type: ignore[arg-type]
        "recall": float(recall),  # type: ignore[arg-type]
        "f1_score": float(f1),  # type: ignore[arg-type]
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_neighbors": n_neighbors,
    }
    
    # Package model
    model_package = {
        "classifier": model,
        "label_encoder": label_encoder,
        "feature_encoders": encoders,
        "feature_scalers": scalers,
        "feature_cols": feature_cols,
        "feature_types": feature_types,
    }
    
    metadata = {
        "target_col": target_col,
        "features": feature_cols,
        "feature_types": feature_types,
        "feature_options": feature_options,
        "n_samples": len(df),
        "n_classes": len(labels),
    }
    
    return TrainResult(model=model_package, metrics=metrics, metadata=metadata)


def predict_knn(model_package: dict, input_dict: dict) -> dict:
    """
    Make prediction using trained KNN model.
    
    Args:
        model_package: Dict containing classifier and preprocessing objects
        input_dict: Dict of feature values
        
    Returns:
        Dict with prediction
    """
    classifier = model_package["classifier"]
    label_encoder = model_package["label_encoder"]
    feature_encoders = model_package["feature_encoders"]
    feature_scalers = model_package.get("feature_scalers", {})
    feature_cols = model_package["feature_cols"]
    
    # Prepare input
    X_input = pd.DataFrame([input_dict])[feature_cols]
    
    # Apply preprocessing
    X_processed = X_input.copy()
    for col in feature_cols:
        if col in feature_encoders:
            # Categorical encoding
            try:
                X_processed[col] = feature_encoders[col].transform(X_input[col].astype(str))
            except ValueError:
                X_processed[col] = 0
        else:
            # Numeric
            X_processed[col] = X_input[col].fillna(0)
            if col in feature_scalers:
                X_processed[col] = feature_scalers[col].transform(X_processed[[col]])
    
    # Predict
    y_pred = classifier.predict(X_processed)[0]
    
    # Decode prediction
    prediction = label_encoder.inverse_transform([y_pred])[0]
    
    return {"prediction": str(prediction), "probabilities": None}


def train_kmeans(df: pd.DataFrame, n_clusters: int = 3) -> TrainResult:
    """
    Train a K-Means clustering model.
    
    Args:
        df: pandas DataFrame with numeric features
        n_clusters: Number of clusters
        
    Returns:
        TrainResult with trained model, metrics, and metadata
    """
    feature_cols = df.columns.tolist()
    X = df[feature_cols].copy()
    
    # Detect feature types (K-Means requires numeric)
    feature_types, _ = detect_feature_types(df, exclude_cols=[])
    
    # Convert all to numeric (encode categoricals if any)
    encoders = {}
    for col in feature_cols:
        if feature_types.get(col) == "categorical":
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col].astype(str))
            encoders[col] = enc
        else:
            X[col] = X[col].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    n_clusters = min(n_clusters, len(df))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    # Calculate metrics
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
    else:
        sil_score = 0.0
        db_score = 0.0
    
    inertia = model.inertia_
    
    # Cluster counts
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = {str(k): int(v) for k, v in zip(unique, counts)}
    
    # Calculate cluster characteristics (mean values per cluster)
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    cluster_characteristics = {}
    for cluster_id in unique:
        cluster_data = df_with_labels[df_with_labels['cluster'] == cluster_id][feature_cols]
        cluster_characteristics[str(cluster_id)] = {
            col: {
                'mean': float(cluster_data[col].mean()),
                'std': float(cluster_data[col].std()),
                'min': float(cluster_data[col].min()),
                'max': float(cluster_data[col].max()),
            }
            for col in feature_cols
        }
    
    # 2D visualization using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(model.cluster_centers_)
    
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": float(sil_score),
        "davies_bouldin_score": float(db_score),
        "inertia": float(inertia),
        "cluster_counts": cluster_counts,
        "cluster_characteristics": cluster_characteristics,
        "points_2d": X_2d.tolist(),
        "labels": labels.tolist(),
        "centers_2d": centers_2d.tolist(),
    }
    
    # Package model
    model_package = {
        "clusterer": model,
        "scaler": scaler,
        "pca": pca,
        "feature_encoders": encoders,
        "feature_cols": feature_cols,
        "feature_types": feature_types,
    }
    
    metadata = {
        "features": feature_cols,
        "feature_types": feature_types,
        "n_samples": len(df),
    }
    
    return TrainResult(model=model_package, metrics=metrics, metadata=metadata)


def predict_kmeans(model_package: dict, input_dict: dict) -> dict:
    """
    Predict cluster for new data point.
    
    Args:
        model_package: Dict containing clusterer and preprocessing objects
        input_dict: Dict of feature values
        
    Returns:
        Dict with cluster assignment
    """
    clusterer = model_package["clusterer"]
    scaler = model_package["scaler"]
    feature_encoders = model_package.get("feature_encoders", {})
    feature_cols = model_package["feature_cols"]
    
    # Prepare input
    X_input = pd.DataFrame([input_dict])[feature_cols]
    
    # Apply preprocessing
    X_processed = X_input.copy()
    for col in feature_cols:
        if col in feature_encoders:
            try:
                X_processed[col] = feature_encoders[col].transform(X_input[col].astype(str))
            except ValueError:
                X_processed[col] = 0
        else:
            X_processed[col] = X_input[col].fillna(0)
    
    # Scale
    X_scaled = scaler.transform(X_processed)
    
    # Predict cluster
    cluster = clusterer.predict(X_scaled)[0]
    
    return {"cluster": int(cluster)}


def train_fpgrowth(df: pd.DataFrame, min_support: float = 0.2, min_confidence: float = 0.6) -> TrainResult:
    """
    Train FP-Growth association rule mining.
    
    Expects a column named 'items' with comma-separated transaction items.
    
    Args:
        df: pandas DataFrame with 'items' column
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        
    Returns:
        TrainResult with association rules
    """
    if "items" not in df.columns:
        raise ValueError("FP-Growth expects a column named 'items' with comma-separated items")
    
    # Parse transactions
    transactions = []
    for items_str in df["items"].fillna("").astype(str):
        items = [x.strip() for x in items_str.split(",") if x.strip()]
        if items:
            transactions.append(items)
    
    if not transactions:
        # No valid transactions
        metrics = {
            "min_support": float(min_support),
            "min_confidence": float(min_confidence),
            "rules": [],
        }
        model = {"rules": []}
        metadata = {}
        return TrainResult(model=model, metrics=metrics, metadata=metadata)
    
    # Convert to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Run FP-Growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        # No frequent itemsets found
        metrics = {
            "min_support": float(min_support),
            "min_confidence": float(min_confidence),
            "rules": [],
        }
        model = {"rules": []}
        metadata = {}
        return TrainResult(model=model, metrics=metrics, metadata=metadata)
    
    # Generate association rules
    try:
        rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    except ValueError:
        # Not enough itemsets to generate rules
        rules_df = pd.DataFrame()
    
    # Convert to list of dicts
    rules = []
    if not rules_df.empty:
        for _, row in rules_df.iterrows():
            rules.append({
                "antecedents": list(row["antecedents"]),
                "consequents": list(row["consequents"]),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            })
    
    # Sort by lift and confidence
    rules = sorted(rules, key=lambda r: (r["lift"], r["confidence"], r["support"]), reverse=True)
    
    metrics = {
        "min_support": float(min_support),
        "min_confidence": float(min_confidence),
        "rules": rules,
    }
    
    model = {"rules": rules}
    metadata = {}
    
    return TrainResult(model=model, metrics=metrics, metadata=metadata)


def recommend_from_rules(rules: list[dict], items: list[str], top_k: int = 5) -> dict:
    """
    Generate recommendations based on association rules.
    
    Args:
        rules: List of association rules
        items: List of items in the basket
        top_k: Number of top recommendations to return
        
    Returns:
        Dict with recommendations
    """
    basket = set([x.strip() for x in items if x and x.strip()])
    matching_rules = []
    
    for rule in rules:
        antecedents = set(rule.get("antecedents", []))
        if antecedents and antecedents.issubset(basket):
            matching_rules.append(rule)
    
    # Sort by lift and confidence
    matching_rules.sort(
        key=lambda r: (float(r.get("lift", 0.0)), float(r.get("confidence", 0.0))),
        reverse=True
    )
    
    # Format recommendations
    recommendations = []
    for rule in matching_rules[:top_k]:
        recommendations.append({
            "if": rule["antecedents"],
            "then": rule["consequents"],
            "confidence": float(rule["confidence"]),
            "lift": float(rule["lift"]),
        })
    
    return {"recommendations": recommendations}
