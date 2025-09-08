# SQLAlchemy models
from sqlalchemy import Column, DateTime, Float, Index, Integer, String
from sqlalchemy.sql import func

from .database import Base


class Prediction(Base):
    """Prediction model for storing phishing detection results."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    img_hash = Column(String, nullable=False, index=True)
    method = Column(String, nullable=False)
    url = Column(String, nullable=False)
    class_ = Column("class", Integer, nullable=False)  # 1 for phishing, 0 for benign
    target = Column(String, nullable=True)
    distance = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    request_id = Column(String, nullable=True)

    # Additional indexes
    __table_args__ = (
        Index("idx_img_hash", "img_hash"),
        Index("idx_created_at", "created_at"),
    )
