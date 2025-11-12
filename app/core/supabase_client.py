"""
Supabase Client Integration

Provides Supabase client for database operations, authentication,
storage, and real-time subscriptions.
"""

from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from functools import lru_cache

from app.config.settings import get_settings

settings = get_settings()


class SupabaseManager:
    """Manager for Supabase operations."""
    
    def __init__(self):
        self.url = settings.supabase_url
        self.key = settings.supabase_key
        self.service_key = settings.supabase_service_key
        self._client: Optional[Client] = None
        self._admin_client: Optional[Client] = None
    
    @property
    def client(self) -> Client:
        """Get Supabase client with anon key (for user operations)."""
        if self._client is None:
            self._client = create_client(self.url, self.key)
        return self._client
    
    @property
    def admin_client(self) -> Client:
        """Get Supabase client with service role key (for admin operations)."""
        if self._admin_client is None:
            self._admin_client = create_client(self.url, self.service_key)
        return self._admin_client
    
    # Authentication Methods
    async def sign_up(self, email: str, password: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Sign up a new user with Supabase Auth.
        
        Args:
            email: User email
            password: User password
            metadata: Additional user metadata
            
        Returns:
            User data and session
        """
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": metadata or {}
                }
            })
            return {
                "user": response.user,
                "session": response.session,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def sign_in(self, email: str, password: str) -> Dict:
        """
        Sign in user with Supabase Auth.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            User data and session
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {
                "user": response.user,
                "session": response.session,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    async def sign_out(self, access_token: str) -> bool:
        """Sign out user."""
        try:
            self.client.auth.sign_out()
            return True
        except Exception as e:
            print(f"Sign out error: {e}")
            return False
    
    async def get_user(self, access_token: str) -> Optional[Dict]:
        """
        Get user from access token.
        
        Args:
            access_token: JWT access token
            
        Returns:
            User data or None
        """
        try:
            response = self.client.auth.get_user(access_token)
            return response.user
        except Exception as e:
            print(f"Get user error: {e}")
            return None
    
    async def refresh_session(self, refresh_token: str) -> Dict:
        """Refresh user session."""
        try:
            response = self.client.auth.refresh_session(refresh_token)
            return {
                "session": response.session,
                "success": True
            }
        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }
    
    # Database Methods
    async def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Select data from Supabase table.
        
        Args:
            table: Table name
            columns: Columns to select
            filters: Filter conditions
            order_by: Order by column
            limit: Limit results
            
        Returns:
            List of records
        """
        try:
            query = self.client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            # Apply ordering
            if order_by:
                query = query.order(order_by)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            print(f"Select error: {e}")
            return []
    
    async def insert(self, table: str, data: Dict[str, Any]) -> Optional[Dict]:
        """
        Insert data into Supabase table.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Inserted record or None
        """
        try:
            response = self.client.table(table).insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Insert error: {e}")
            return None
    
    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Update data in Supabase table.
        
        Args:
            table: Table name
            data: Data to update
            filters: Filter conditions
            
        Returns:
            Updated record or None
        """
        try:
            query = self.client.table(table).update(data)
            
            for key, value in filters.items():
                query = query.eq(key, value)
            
            response = query.execute()
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Update error: {e}")
            return None
    
    async def delete(self, table: str, filters: Dict[str, Any]) -> bool:
        """
        Delete data from Supabase table.
        
        Args:
            table: Table name
            filters: Filter conditions
            
        Returns:
            True if successful
        """
        try:
            query = self.client.table(table).delete()
            
            for key, value in filters.items():
                query = query.eq(key, value)
            
            query.execute()
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    # Storage Methods
    async def upload_file(
        self,
        bucket: str,
        path: str,
        file_data: bytes,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload file to Supabase Storage.
        
        Args:
            bucket: Storage bucket name
            path: File path in bucket
            file_data: File bytes
            content_type: MIME type
            
        Returns:
            Public URL or None
        """
        try:
            response = self.client.storage.from_(bucket).upload(
                path,
                file_data,
                {"content-type": content_type} if content_type else {}
            )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(path)
            return public_url
            
        except Exception as e:
            print(f"Upload error: {e}")
            return None
    
    async def download_file(self, bucket: str, path: str) -> Optional[bytes]:
        """Download file from Supabase Storage."""
        try:
            response = self.client.storage.from_(bucket).download(path)
            return response
        except Exception as e:
            print(f"Download error: {e}")
            return None
    
    async def delete_file(self, bucket: str, path: str) -> bool:
        """Delete file from Supabase Storage."""
        try:
            self.client.storage.from_(bucket).remove([path])
            return True
        except Exception as e:
            print(f"Delete file error: {e}")
            return False
    
    async def get_public_url(self, bucket: str, path: str) -> str:
        """Get public URL for file."""
        return self.client.storage.from_(bucket).get_public_url(path)
    
    # Real-time Methods
    def subscribe_to_table(self, table: str, callback):
        """
        Subscribe to real-time changes on a table.
        
        Args:
            table: Table name
            callback: Callback function for changes
        """
        return self.client.table(table).on('*', callback).subscribe()
    
    # RPC Methods
    async def call_function(self, function_name: str, params: Optional[Dict] = None) -> Any:
        """
        Call Supabase Edge Function or Database Function.
        
        Args:
            function_name: Function name
            params: Function parameters
            
        Returns:
            Function result
        """
        try:
            response = self.client.rpc(function_name, params or {}).execute()
            return response.data
        except Exception as e:
            print(f"RPC error: {e}")
            return None


@lru_cache()
def get_supabase() -> SupabaseManager:
    """Get cached Supabase manager instance."""
    return SupabaseManager()


# Dependency for FastAPI
def get_supabase_client() -> SupabaseManager:
    """FastAPI dependency to get Supabase client."""
    return get_supabase()