"""
Base Repository Pattern

Provides base repository class and common database operations
for data access layer abstraction.
"""

from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository class with common CRUD operations."""
    
    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository with model and database session.
        
        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
    
    def get(self, id: UUID) -> Optional[ModelType]:
        """
        Get a single record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            Model instance or None
        """
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_or_404(self, id: UUID) -> ModelType:
        """
        Get a single record by ID or raise 404.
        
        Args:
            id: Record ID
            
        Returns:
            Model instance
            
        Raises:
            HTTPException: If record not found
        """
        record = self.get(id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{self.model.__name__} not found"
            )
        return record
    
    def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            
        Returns:
            List of model instances
        """
        query = self.db.query(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
        
        return query.offset(skip).limit(limit).all()
    
    def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """
        Create a new record.
        
        Args:
            obj_in: Dictionary of field values
            
        Returns:
            Created model instance
            
        Raises:
            HTTPException: If creation fails due to constraint violation
        """
        try:
            db_obj = self.model(**obj_in)
            self.db.add(db_obj)
            self.db.commit()
            self.db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create {self.model.__name__}: {str(e)}"
            )
    
    def update(self, id: UUID, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an existing record.
        
        Args:
            id: Record ID
            obj_in: Dictionary of field values to update
            
        Returns:
            Updated model instance or None if not found
        """
        db_obj = self.get(id)
        if not db_obj:
            return None
        
        try:
            for key, value in obj_in.items():
                if hasattr(db_obj, key):
                    setattr(db_obj, key, value)
            
            self.db.commit()
            self.db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update {self.model.__name__}: {str(e)}"
            )
    
    def delete(self, id: UUID) -> bool:
        """
        Delete a record by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if deleted, False if not found
        """
        db_obj = self.get(id)
        if not db_obj:
            return False
        
        self.db.delete(db_obj)
        self.db.commit()
        return True
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records with optional filtering.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of matching records
        """
        query = self.db.query(self.model)
        
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
        
        return query.count()
    
    def exists(self, id: UUID) -> bool:
        """
        Check if a record exists by ID.
        
        Args:
            id: Record ID
            
        Returns:
            True if record exists
        """
        return self.db.query(self.model).filter(self.model.id == id).first() is not None
    
    def get_by_field(self, field_name: str, field_value: Any) -> Optional[ModelType]:
        """
        Get a single record by field value.
        
        Args:
            field_name: Field name to search by
            field_value: Field value to match
            
        Returns:
            Model instance or None
        """
        if not hasattr(self.model, field_name):
            return None
        
        return self.db.query(self.model).filter(
            getattr(self.model, field_name) == field_value
        ).first()
    
    def get_multi_by_field(
        self,
        field_name: str,
        field_value: Any,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """
        Get multiple records by field value.
        
        Args:
            field_name: Field name to search by
            field_value: Field value to match
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of model instances
        """
        if not hasattr(self.model, field_name):
            return []
        
        return self.db.query(self.model).filter(
            getattr(self.model, field_name) == field_value
        ).offset(skip).limit(limit).all()
    
    def get_all(self, skip: int = 0, limit: int = 1000) -> List[ModelType]:
        """
        Get all records with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of all model instances
        """
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def get_multi_by_ids(self, ids: List[UUID]) -> List[ModelType]:
        """
        Get multiple records by list of IDs.
        
        Args:
            ids: List of record IDs
            
        Returns:
            List of model instances
        """
        return self.db.query(self.model).filter(self.model.id.in_(ids)).all()