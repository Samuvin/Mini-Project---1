"""Prediction result model operations for MongoDB."""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from bson.objectid import ObjectId

from webapp.db import get_db

logger = logging.getLogger(__name__)


def _predictions_collection():
    """Return the ``predictions`` collection and ensure indexes exist."""
    db = get_db()
    collection = db["predictions"]
    # Ensure indexes for efficient queries
    collection.create_index("user_id")
    collection.create_index([("user_id", 1), ("created_at", -1)])
    return collection


def save_prediction(user_id: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a prediction result to the database.

    Args:
        user_id: The user's MongoDB ObjectId (as string or ObjectId).
        result_data: Dictionary containing prediction result data:
            - prediction (int): 0 or 1
            - prediction_label (str): "Healthy" or "Parkinson's Disease"
            - confidence (float): 0.0 to 1.0
            - probabilities (dict): {"Healthy": float, "Parkinson's Disease": float}
            - modalities_used (list): List of modality strings
            - model_type (str): "dl" or "sklearn"

    Returns:
        dict: The saved prediction document with _id.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    prediction_doc = {
        "user_id": user_id,
        "prediction": result_data.get("prediction", 0),
        "prediction_label": result_data.get("prediction_label", "Unknown"),
        "confidence": result_data.get("confidence", 0.0),
        "probabilities": result_data.get("probabilities", {}),
        "modalities_used": result_data.get("modalities_used", []),
        "model_type": result_data.get("model_type", "sklearn"),
        "created_at": datetime.now(timezone.utc),
    }

    result = _predictions_collection().insert_one(prediction_doc)
    prediction_doc["_id"] = result.inserted_id
    logger.info("Prediction saved for user %s: %s (%.2f%% confidence)", 
                user_id, prediction_doc["prediction_label"], 
                prediction_doc["confidence"] * 100)

    # Convert ObjectId to string for JSON serialization
    prediction_doc["_id"] = str(prediction_doc["_id"])
    prediction_doc["user_id"] = str(prediction_doc["user_id"])
    # Ensure UTC timezone indicator ('Z') is included
    created_at_iso = prediction_doc["created_at"].isoformat()
    if not created_at_iso.endswith('Z') and not ('+' in created_at_iso or created_at_iso.count('-') > 2):
        created_at_iso = created_at_iso + 'Z'
    prediction_doc["created_at"] = created_at_iso

    return prediction_doc


def find_by_user_id(user_id: str, limit: Optional[int] = None, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Get predictions for a specific user.

    Args:
        user_id: The user's MongoDB ObjectId (as string or ObjectId).
        limit: Maximum number of results to return (None for no limit).
        skip: Number of results to skip (for pagination).

    Returns:
        list: List of prediction documents, ordered by created_at descending.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    query = {"user_id": user_id}
    cursor = _predictions_collection().find(query).sort("created_at", -1)

    if skip > 0:
        cursor = cursor.skip(skip)
    if limit is not None:
        cursor = cursor.limit(limit)

    results = list(cursor)
    
    # Convert ObjectIds and datetimes for JSON serialization
    for doc in results:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            created_at_iso = doc["created_at"].isoformat()
            # Ensure UTC timezone indicator ('Z') is included
            if not created_at_iso.endswith('Z') and not ('+' in created_at_iso or created_at_iso.count('-') > 2):
                created_at_iso = created_at_iso + 'Z'
            doc["created_at"] = created_at_iso

    return results


def search_predictions(user_id: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search predictions with filters.

    Args:
        user_id: The user's MongoDB ObjectId (as string or ObjectId).
        filters: Dictionary with optional filters:
            - date_from: ISO datetime string (inclusive)
            - date_to: ISO datetime string (inclusive)
            - result: "Healthy" or "Parkinson's Disease" or None
            - min_confidence: float (0.0-1.0)
            - max_confidence: float (0.0-1.0)
            - limit: int (default: 50)
            - skip: int (default: 0)

    Returns:
        list: List of prediction documents matching the filters.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    query = {"user_id": user_id}

    # Date range filter
    date_filter = {}
    if filters.get("date_from"):
        try:
            date_from = datetime.fromisoformat(filters["date_from"].replace("Z", "+00:00"))
            date_filter["$gte"] = date_from
        except (ValueError, AttributeError):
            logger.warning("Invalid date_from format: %s", filters.get("date_from"))
    
    if filters.get("date_to"):
        try:
            date_to = datetime.fromisoformat(filters["date_to"].replace("Z", "+00:00"))
            date_filter["$lte"] = date_to
        except (ValueError, AttributeError):
            logger.warning("Invalid date_to format: %s", filters.get("date_to"))
    
    if date_filter:
        query["created_at"] = date_filter

    # Result filter
    if filters.get("result"):
        query["prediction_label"] = filters["result"]

    # Confidence range filter
    confidence_filter = {}
    if filters.get("min_confidence") is not None:
        confidence_filter["$gte"] = float(filters["min_confidence"])
    if filters.get("max_confidence") is not None:
        confidence_filter["$lte"] = float(filters["max_confidence"])
    
    if confidence_filter:
        query["confidence"] = confidence_filter

    # Pagination
    limit = filters.get("limit", 50)
    skip = filters.get("skip", 0)

    cursor = _predictions_collection().find(query).sort("created_at", -1)
    
    if skip > 0:
        cursor = cursor.skip(skip)
    if limit is not None:
        cursor = cursor.limit(limit)

    results = list(cursor)
    
    # Convert ObjectIds and datetimes for JSON serialization
    for doc in results:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()

    return results


def find_by_id(result_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Find a prediction by ID, ensuring it belongs to the user.

    Args:
        result_id: The prediction's MongoDB ObjectId (as string).
        user_id: The user's MongoDB ObjectId (as string or ObjectId).

    Returns:
        dict or None: The prediction document, or None if not found or doesn't belong to user.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    try:
        result_id_obj = ObjectId(result_id)
    except Exception:
        return None

    doc = _predictions_collection().find_one({
        "_id": result_id_obj,
        "user_id": user_id
    })

    if doc:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            created_at_iso = doc["created_at"].isoformat()
            # Ensure UTC timezone indicator ('Z') is included
            if not created_at_iso.endswith('Z') and not ('+' in created_at_iso or created_at_iso.count('-') > 2):
                created_at_iso = created_at_iso + 'Z'
            doc["created_at"] = created_at_iso

    return doc


def count_by_user_id(user_id: str) -> int:
    """
    Count total predictions for a user.

    Args:
        user_id: The user's MongoDB ObjectId (as string or ObjectId).

    Returns:
        int: Total number of predictions.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    return _predictions_collection().count_documents({"user_id": user_id})
