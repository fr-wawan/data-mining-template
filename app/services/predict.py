from __future__ import annotations

import joblib
from sqlalchemy.orm import Session

from app.models.model_artifact import ModelArtifact
from app.models.prediction import PredictionHistory
from app.services.ml.algorithms import predict_kmeans, predict_knn, predict_naive_bayes, recommend_from_rules


def run_prediction(db: Session, user_id: int, model_artifact: ModelArtifact, input_dict: dict) -> PredictionHistory:
    """
    Execute prediction using a trained model and save the result to history.
    
    Args:
        db: Database session
        user_id: ID of the user making the prediction
        model_artifact: The trained model artifact to use
        input_dict: Input data for prediction
        
    Returns:
        PredictionHistory record with input and output
        
    Raises:
        ValueError: If algorithm is not supported
    """
    # Load model from joblib file
    model_package = joblib.load(model_artifact.file_path)
    algo = model_artifact.algorithm.lower()

    if algo in ("naive_bayes", "knn"):
        if algo == "naive_bayes":
            out = predict_naive_bayes(model_package, input_dict)
        else:
            out = predict_knn(model_package, input_dict)
    elif algo == "kmeans":
        out = predict_kmeans(model_package, input_dict)
    elif algo == "fpgrowth":
        items = input_dict.get("items", "")
        items_list = [x.strip() for x in str(items).split(",") if x.strip()]
        rules = model_package.get("rules", [])
        out = recommend_from_rules(rules, items_list)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    hist = PredictionHistory(user_id=user_id, model_id=model_artifact.id, input_json=input_dict, output_json=out)
    db.add(hist)
    db.commit()
    db.refresh(hist)
    return hist
