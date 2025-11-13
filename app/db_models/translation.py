"""
Translation Database Models

SQLAlchemy models for translation operations and history.
"""

from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

from app.core.database import Base


class Translation(Base):
    """Translation history model."""
    
    __tablename__ = "translations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Translation details
    source_type = Column(String(50), nullable=False)  # speech, text, sign
    target_type = Column(String(50), nullable=False)  # speech, text, sign
    source_content = Column(Text, nullable=True)
    target_content = Column(Text, nullable=True)
    
    # Metadata
    confidence_score = Column(Float, default=0.0)
    processing_time = Column(Float, nullable=True)  # in seconds
    metadata = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Translation(id={self.id}, {self.source_type}->{self.target_type})>"


class SignSequence(Base):
    """Sign sequence for animated demonstrations."""
    
    __tablename__ = "sign_sequences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    translation_id = Column(UUID(as_uuid=True), ForeignKey("translations.id"), nullable=True)
    
    # Sequence details
    original_text = Column(Text, nullable=False)
    sign_ids = Column(JSONB, nullable=False)  # Ordered list of sign IDs
    transitions = Column(JSONB, default=list)  # Transition timing and effects
    
    # Metadata
    estimated_duration = Column(Float, nullable=True)  # in seconds
    grammar_applied = Column(String(50), default="harmonized_gsl")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SignSequence(id={self.id}, signs={len(self.sign_ids)})>"
