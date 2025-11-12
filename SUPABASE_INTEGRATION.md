# Supabase Integration Guide

## Overview
This guide explains how to integrate the GSL Backend with Supabase for database, authentication, storage, and real-time features.

## Why Supabase?

Supabase provides:
- **PostgreSQL Database**: Fully managed PostgreSQL with automatic backups
- **Built-in Authentication**: User management, JWT tokens, OAuth providers
- **Storage**: File storage with CDN for videos and images
- **Real-time**: WebSocket subscriptions for live updates
- **Row Level Security**: Database-level security policies
- **Auto-generated APIs**: REST and GraphQL APIs
- **Edge Functions**: Serverless functions for custom logic

## Setup Steps

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Create a new organization (if needed)
4. Create a new project:
   - **Name**: `gsl-backend`
   - **Database Password**: Choose a strong password
   - **Region**: Select closest to Ghana (e.g., `eu-west-1`)

### 2. Get Supabase Credentials

After project creation, go to **Settings > API**:

```bash
# Project URL
SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co

# Anon/Public Key (safe to use in client-side code)
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Service Role Key (NEVER expose in client-side code)
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Database Connection String
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxxx.supabase.co:5432/postgres
```

### 3. Update Environment Variables

Update your `.env` file:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-role-key
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres
USE_SUPABASE_AUTH=true
```

## Database Schema Setup

### Option 1: Using Supabase Dashboard (Recommended)

1. Go to **Table Editor** in Supabase Dashboard
2. Create tables using the SQL Editor or Table Editor UI
3. Enable Row Level Security (RLS) for each table
4. Create policies for access control

### Option 2: Using SQL Migrations

Create a migration file in Supabase:

```sql
-- Create users table (extends Supabase auth.users)
CREATE TABLE public.learner_profiles (
  id UUID REFERENCES auth.users(id) PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  full_name TEXT,
  age_group TEXT,
  learning_level TEXT DEFAULT 'beginner',
  preferred_language TEXT DEFAULT 'english',
  accessibility_needs JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create learning progress table
CREATE TABLE public.learning_progress (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  total_lessons_completed INTEGER DEFAULT 0,
  current_level INTEGER DEFAULT 1,
  completed_lessons JSONB DEFAULT '[]'::jsonb,
  achievements JSONB DEFAULT '[]'::jsonb,
  last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create GSL signs table
CREATE TABLE public.gsl_signs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  sign_name TEXT NOT NULL,
  description TEXT,
  category_id UUID REFERENCES public.sign_categories(id),
  difficulty_level INTEGER DEFAULT 1,
  video_url TEXT,
  thumbnail_url TEXT,
  related_signs JSONB DEFAULT '[]'::jsonb,
  usage_examples JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create sign categories table
CREATE TABLE public.sign_categories (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  parent_category_id UUID REFERENCES public.sign_categories(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create lessons table
CREATE TABLE public.lessons (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT,
  level INTEGER DEFAULT 1,
  category TEXT,
  sequence_order INTEGER,
  signs_covered JSONB DEFAULT '[]'::jsonb,
  estimated_duration INTEGER,
  prerequisites JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create media files table
CREATE TABLE public.media_files (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_url TEXT,
  thumbnail_url TEXT,
  file_size BIGINT,
  content_type TEXT,
  file_hash TEXT,
  uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create sign recognition results table
CREATE TABLE public.sign_recognitions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id),
  recognized_sign_id UUID REFERENCES public.gsl_signs(id),
  confidence_score FLOAT,
  processing_time FLOAT,
  status TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.learner_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.learning_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.media_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sign_recognitions ENABLE ROW LEVEL SECURITY;

-- Create RLS Policies
-- Users can read their own profile
CREATE POLICY "Users can view own profile"
  ON public.learner_profiles FOR SELECT
  USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
  ON public.learner_profiles FOR UPDATE
  USING (auth.uid() = id);

-- Users can read their own progress
CREATE POLICY "Users can view own progress"
  ON public.learning_progress FOR SELECT
  USING (auth.uid() = user_id);

-- Users can update their own progress
CREATE POLICY "Users can update own progress"
  ON public.learning_progress FOR UPDATE
  USING (auth.uid() = user_id);

-- Everyone can read GSL signs (public data)
CREATE POLICY "Anyone can view GSL signs"
  ON public.gsl_signs FOR SELECT
  TO authenticated, anon
  USING (true);

-- Everyone can read lessons (public data)
CREATE POLICY "Anyone can view lessons"
  ON public.lessons FOR SELECT
  TO authenticated, anon
  USING (true);

-- Users can read their own media files
CREATE POLICY "Users can view own media"
  ON public.media_files FOR SELECT
  USING (auth.uid() = user_id);

-- Users can insert their own media files
CREATE POLICY "Users can upload media"
  ON public.media_files FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Create indexes for performance
CREATE INDEX idx_learner_profiles_username ON public.learner_profiles(username);
CREATE INDEX idx_learning_progress_user_id ON public.learning_progress(user_id);
CREATE INDEX idx_gsl_signs_category ON public.gsl_signs(category_id);
CREATE INDEX idx_gsl_signs_name ON public.gsl_signs(sign_name);
CREATE INDEX idx_lessons_level ON public.lessons(level);
CREATE INDEX idx_media_files_user_id ON public.media_files(user_id);
```

## Storage Setup

### Create Storage Buckets

1. Go to **Storage** in Supabase Dashboard
2. Create buckets:
   - `sign-videos`: For GSL sign demonstration videos
   - `user-uploads`: For user-uploaded gesture videos
   - `thumbnails`: For video thumbnails
   - `lesson-media`: For lesson content media

### Configure Bucket Policies

```sql
-- Allow authenticated users to upload to user-uploads
CREATE POLICY "Users can upload own files"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'user-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow authenticated users to read their own uploads
CREATE POLICY "Users can view own uploads"
ON storage.objects FOR SELECT
TO authenticated
USING (bucket_id = 'user-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Allow everyone to read sign videos (public)
CREATE POLICY "Anyone can view sign videos"
ON storage.objects FOR SELECT
TO authenticated, anon
USING (bucket_id = 'sign-videos');
```

## Authentication Setup

### Enable Email Authentication

1. Go to **Authentication > Providers**
2. Enable **Email** provider
3. Configure email templates (optional)

### Configure JWT Settings

1. Go to **Settings > API**
2. Note the JWT Secret (used for token verification)
3. Set JWT expiry time (default: 1 hour)

## Code Integration

### Using Supabase Client

```python
from app.core.supabase_client import get_supabase

# Get Supabase client
supabase = get_supabase()

# Sign up user
result = await supabase.sign_up(
    email="user@example.com",
    password="secure_password",
    metadata={"full_name": "John Doe"}
)

# Sign in user
result = await supabase.sign_in(
    email="user@example.com",
    password="secure_password"
)

# Query data
signs = await supabase.select(
    table="gsl_signs",
    columns="*",
    filters={"difficulty_level": 1},
    limit=10
)

# Insert data
new_sign = await supabase.insert(
    table="gsl_signs",
    data={
        "sign_name": "hello",
        "description": "Greeting sign",
        "difficulty_level": 1
    }
)

# Upload file
url = await supabase.upload_file(
    bucket="user-uploads",
    path=f"{user_id}/gesture.mp4",
    file_data=video_bytes,
    content_type="video/mp4"
)
```

### Using with FastAPI Endpoints

```python
from fastapi import Depends
from app.core.supabase_client import get_supabase_client, SupabaseManager

@router.post("/register")
async def register_user(
    user_data: UserCreate,
    supabase: SupabaseManager = Depends(get_supabase_client)
):
    # Register with Supabase Auth
    result = await supabase.sign_up(
        email=user_data.email,
        password=user_data.password,
        metadata={
            "username": user_data.username,
            "full_name": user_data.full_name
        }
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    # Create learner profile
    profile = await supabase.insert(
        table="learner_profiles",
        data={
            "id": result["user"].id,
            "username": user_data.username,
            "full_name": user_data.full_name
        }
    )
    
    return {
        "user": result["user"],
        "access_token": result["session"].access_token
    }
```

## Real-time Features

### Subscribe to Table Changes

```python
def handle_new_sign(payload):
    print(f"New sign added: {payload}")

# Subscribe to new GSL signs
supabase.subscribe_to_table("gsl_signs", handle_new_sign)
```

## Edge Functions (Optional)

Create serverless functions for:
- AI model inference
- Video processing
- Complex business logic

Example Edge Function:

```typescript
// supabase/functions/recognize-gesture/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"

serve(async (req) => {
  const { video_url } = await req.json()
  
  // Call AI model
  const result = await recognizeGesture(video_url)
  
  return new Response(
    JSON.stringify(result),
    { headers: { "Content-Type": "application/json" } }
  )
})
```

## Migration from PostgreSQL to Supabase

### Data Migration

1. Export existing PostgreSQL data:
```bash
pg_dump -h localhost -U gsl_user -d gsl_db > backup.sql
```

2. Import to Supabase:
```bash
psql -h db.your-project.supabase.co -U postgres -d postgres < backup.sql
```

### Code Migration Checklist

- [x] Update `requirements.txt` with Supabase SDK
- [x] Create `supabase_client.py` for Supabase operations
- [x] Update `settings.py` with Supabase credentials
- [x] Update `.env.example` with Supabase variables
- [ ] Create database schema in Supabase
- [ ] Set up storage buckets
- [ ] Configure RLS policies
- [ ] Update service layer to use Supabase client
- [ ] Test authentication flow
- [ ] Test file upload/download
- [ ] Migrate existing data (if any)

## Next Steps

1. **Create Supabase Project** and get credentials
2. **Set up database schema** using SQL migrations
3. **Configure storage buckets** for media files
4. **Update services** to use Supabase client
5. **Test authentication** with Supabase Auth
6. **Implement file upload** to Supabase Storage
7. **Add real-time features** for live updates
8. **Deploy Edge Functions** for AI processing (optional)

## Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)
- [Storage Guide](https://supabase.com/docs/guides/storage)
- [Edge Functions Guide](https://supabase.com/docs/guides/functions)