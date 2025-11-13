"""
Role-Based Access Control (RBAC)

Provides role-based access control decorators and utilities.
"""

from typing import List
from functools import wraps
from fastapi import HTTPException, status, Depends

from app.db_models.user import User, UserRole
from app.core.security import get_current_active_user


def require_roles(allowed_roles: List[UserRole]):
    """
    Decorator to require specific user roles for endpoint access.
    
    Args:
        allowed_roles: List of allowed user roles
        
    Returns:
        Dependency function that checks user role
        
    Raises:
        HTTPException: If user doesn't have required role
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    
    return role_checker


def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency to require admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_teacher_or_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Dependency to require teacher or admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User if teacher or admin
        
    Raises:
        HTTPException: If user is not teacher or admin
    """
    if current_user.role not in [UserRole.teacher, UserRole.admin]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher or admin access required"
        )
    return current_user


def check_user_permission(user: User, resource_owner_id: str) -> bool:
    """
    Check if user has permission to access a resource.
    
    Args:
        user: Current user
        resource_owner_id: ID of resource owner
        
    Returns:
        True if user has permission
    """
    # Admin can access everything
    if user.role == UserRole.admin:
        return True
    
    # User can access their own resources
    if str(user.id) == resource_owner_id:
        return True
    
    # Teacher can access learner resources
    if user.role == UserRole.teacher:
        return True
    
    return False


def require_self_or_admin(resource_user_id: str):
    """
    Dependency to require user to be accessing their own resource or be admin.
    
    Args:
        resource_user_id: ID of the resource owner
        
    Returns:
        Dependency function
    """
    def permission_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not check_user_permission(current_user, resource_user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only access your own resources."
            )
        return current_user
    
    return permission_checker
