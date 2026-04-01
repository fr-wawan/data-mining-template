from __future__ import annotations

import os
from datetime import datetime

import joblib
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.models.model_artifact import ModelArtifact
from app.services.dataio import load_table
from app.services.ml.algorithms import train_fpgrowth, train_kmeans, train_knn, train_naive_bayes
from app.services.storage import ensure_dirs


def _artifact_path(algorithm: str, dataset_id: int) -> str:
    """Generate a unique file path for saving model artifact."""
    ensure_dirs()
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    fname = f"{algorithm}_ds{dataset_id}_{ts}.joblib"
    return os.path.join(settings.MODEL_DIR, fname)


def train_model(db: Session, dataset: Dataset, algorithm: str, params: dict) -> ModelArtifact:
    """
    Train a machine learning model on the given dataset.
    
    Args:
        db: Database session
        dataset: Dataset to train on
        algorithm: Algorithm name (naive_bayes, knn, kmeans, fpgrowth)
        params: Training parameters (e.g., target_col, n_neighbors, n_clusters)
        
    Returns:
        ModelArtifact with trained model metadata and metrics
        
    Raises:
        ValueError: If algorithm is not supported
    """
    df = load_table(os.path.join(settings.DATASET_DIR, dataset.filename))

    algorithm = algorithm.lower()
    if algorithm == "naive_bayes":
        target_col = params.get("target_col", "target")
        res = train_naive_bayes(df, target_col=target_col)
    elif algorithm == "knn":
        target_col = params.get("target_col", "target")
        k = int(params.get("n_neighbors", 5))
        res = train_knn(df, target_col=target_col, n_neighbors=k)
    elif algorithm == "kmeans":
        k = int(params.get("n_clusters", 3))
        res = train_kmeans(df, n_clusters=k)
    elif algorithm == "fpgrowth":
        res = train_fpgrowth(
            df,
            min_support=float(params.get("min_support", 0.2)),
            min_confidence=float(params.get("min_confidence", 0.6)),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Save model using joblib
    out_path = _artifact_path(algorithm, dataset.id)
    joblib.dump(res.model, out_path)

    artifact = ModelArtifact(
        name=f"{algorithm.upper()} - {dataset.name}",
        algorithm=algorithm,
        dataset_id=dataset.id,
        file_path=out_path,
        metadata_json=res.metadata,
        metrics_json=res.metrics,
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact
