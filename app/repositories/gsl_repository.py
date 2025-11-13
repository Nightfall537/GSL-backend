"""
GSL Repository

Repository classes for GSL-related database operations.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.core.repository import BaseRepository
from app.db_models.gsl import GSLSign, SignCategory


class SignCategoryRepository(BaseRepository[SignCategory]):
    """Repository for SignCategory model operations."""
    
    def __init__(self, db: Session):
        super().__init__(SignCategory, db)
    
    def get_by_name(self, name: str) -> Optional[SignCategory]:
        """
        Get category by name.
        
        Args:
            name: Category name
            
        Returns:
            SignCategory instance or None
        """
        return self.db.query(SignCategory).filter(
            SignCategory.name == name.lower()
        ).first()
    
    def get_root_categories(self) -> List[SignCategory]:
        """
        Get all root categories (no parent).
        
        Returns:
            List of root categories
        """
        return self.db.query(SignCategory).filter(
            SignCategory.parent_category_id.is_(None)
        ).all()
    
    def get_subcategories(self, parent_id: UUID) -> List[SignCategory]:
        """
        Get subcategories of a parent category.
        
        Args:
            parent_id: Parent category ID
            
        Returns:
            List of subcategories
        """
        return self.db.query(SignCategory).filter(
            SignCategory.parent_category_id == parent_id
        ).all()


class GSLSignRepository(BaseRepository[GSLSign]):
    """Repository for GSLSign model operations."""
    
    def __init__(self, db: Session):
        super().__init__(GSLSign, db)
    
    def search_signs(
        self,
        query: Optional[str] = None,
        category_id: Optional[UUID] = None,
        difficulty_level: Optional[int] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[GSLSign]:
        """
        Search GSL signs with filters.
        
        Args:
            query: Search query for sign name or description
            category_id: Filter by category ID
            difficulty_level: Filter by difficulty level
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching GSL signs
        """
        db_query = self.db.query(GSLSign)
        
        # Text search
        if query:
            search_term = f"%{query.lower()}%"
            db_query = db_query.filter(
                or_(
                    GSLSign.sign_name.ilike(search_term),
                    GSLSign.description.ilike(search_term)
                )
            )
        
        # Category filter
        if category_id:
            db_query = db_query.filter(GSLSign.category_id == category_id)
        
        # Difficulty filter
        if difficulty_level:
            db_query = db_query.filter(GSLSign.difficulty_level == difficulty_level)
        
        return db_query.offset(skip).limit(limit).all()
    
    def get_by_name(self, sign_name: str) -> Optional[GSLSign]:
        """
        Get sign by exact name match.
        
        Args:
            sign_name: Sign name to search for
            
        Returns:
            GSLSign instance or None
        """
        return self.db.query(GSLSign).filter(
            GSLSign.sign_name.ilike(sign_name)
        ).first()
    
    def get_by_category(
        self,
        category_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[GSLSign]:
        """
        Get signs by category.
        
        Args:
            category_id: Category ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of signs in category
        """
        return self.db.query(GSLSign).filter(
            GSLSign.category_id == category_id
        ).offset(skip).limit(limit).all()
    
    def get_by_difficulty(
        self,
        difficulty_level: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[GSLSign]:
        """
        Get signs by difficulty level.
        
        Args:
            difficulty_level: Difficulty level (1-3)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of signs at difficulty level
        """
        return self.db.query(GSLSign).filter(
            GSLSign.difficulty_level == difficulty_level
        ).offset(skip).limit(limit).all()
    
    def get_related_signs(self, sign_id: UUID) -> List[GSLSign]:
        """
        Get signs related to a specific sign.
        
        Args:
            sign_id: Sign ID to find related signs for
            
        Returns:
            List of related signs
        """
        sign = self.get(sign_id)
        if not sign or not sign.related_signs:
            return []
        
        return self.db.query(GSLSign).filter(
            GSLSign.id.in_(sign.related_signs)
        ).all()
    
    def get_signs_with_video(self, skip: int = 0, limit: int = 100) -> List[GSLSign]:
        """
        Get signs that have video demonstrations.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of signs with videos
        """
        return self.db.query(GSLSign).filter(
            GSLSign.video_url.isnot(None)
        ).offset(skip).limit(limit).all()
    
    def get_popular_signs(self, limit: int = 10) -> List[GSLSign]:
        """
        Get popular signs (placeholder - would need usage tracking).
        
        Args:
            limit: Maximum number of signs to return
            
        Returns:
            List of popular signs
        """
        # For now, return signs with difficulty level 1 (beginner-friendly)
        return self.db.query(GSLSign).filter(
            GSLSign.difficulty_level == 1
        ).limit(limit).all()
    
    def count_by_category(self, category_id: UUID) -> int:
        """
        Count signs in a category.
        
        Args:
            category_id: Category ID
            
        Returns:
            Number of signs in category
        """
        return self.db.query(GSLSign).filter(
            GSLSign.category_id == category_id
        ).count()
    
    def get_signs_by_ids(self, sign_ids: List[UUID]) -> List[GSLSign]:
        """
        Get multiple signs by their IDs.
        
        Args:
            sign_ids: List of sign IDs
            
        Returns:
            List of GSL signs
        """
        return self.db.query(GSLSign).filter(
            GSLSign.id.in_(sign_ids)
        ).all()
    
    def add_related_sign(self, sign_id: UUID, related_sign_id: UUID) -> bool:
        """
        Add a related sign relationship.
        
        Args:
            sign_id: Main sign ID
            related_sign_id: Related sign ID to add
            
        Returns:
            True if successful
        """
        sign = self.get(sign_id)
        if not sign:
            return False
        
        related_signs = sign.related_signs or []
        if related_sign_id not in related_signs:
            related_signs.append(related_sign_id)
            self.update(sign_id, {"related_signs": related_signs})
        
        return True