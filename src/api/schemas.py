from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Schema for prediction request."""

    image: str
    url: str


class PredictionCreate(BaseModel):
    """Schema for creating a prediction."""

    img_hash: str
    method: str
    url: str
    class_: int
    target: Optional[str] = None
    distance: Optional[float] = None


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    id: int
    img_hash: str
    method: str
    url: str
    class_: int
    target: Optional[str] = None
    distance: Optional[float] = None
    created_at: datetime
    request_id: Optional[str] = None

    model_config = {"from_attributes": True}


class PredictionListResponse(BaseModel):
    """Schema for list of predictions response."""

    predictions: List[PredictionResponse]
    total: int
