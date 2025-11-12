# Quick Start Guide - GSL Backend with Supabase

## Prerequisites
- Python 3.11+
- Supabase account
- Git

## 1. Clone and Setup (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd gsl-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Create Supabase Project (10 minutes)

1. Go to [supabase.com](https://supabase.com) and sign up
2. Click "New Project"
3. Fill in:
   - **Name**: `gsl-backend`
   - **Database Password**: (save this!)
   - **Region**: `eu-west-1` (closest to Ghana)
4. Wait for project to be created (~2 minutes)

## 3. Get Supabase Credentials (2 minutes)

1. Go to **Settings > API** in your Supabase project
2. Copy these values:
   - **Project URL**
   - **anon public key**
   - **service_role key** (keep secret!)
3. Go to **Settings > Database**
4. Copy **Connection string** (URI format)

## 4. Configure Environment (3 minutes)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your Supabase credentials
```

Update `.env`:
```bash
# Supabase Configuration
SUPABASE_URL=https://xxxxxxxxxxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.xxxxxxxxxxxxx.supabase.co:5432/postgres
USE_SUPABASE_AUTH=true

# Application
SECRET_KEY=your-secret-key-change-this-to-something-random
DEBUG=true
ENVIRONMENT=development
```

## 5. Create Database Schema (15 minutes)

1. Go to **SQL Editor** in Supabase Dashboard
2. Copy the SQL from `SUPABASE_INTEGRATION.md` (Database Schema Setup section)
3. Paste and run the SQL
4. Verify tables are created in **Table Editor**

## 6. Create Storage Buckets (5 minutes)

1. Go to **Storage** in Supabase Dashboard
2. Create these buckets:
   - `sign-videos` (public)
   - `user-uploads` (private)
   - `thumbnails` (public)
   - `lesson-media` (public)

## 7. Test Connection (2 minutes)

```bash
# Test Supabase connection
python -c "from app.core.supabase_client import get_supabase; client = get_supabase(); print('✅ Supabase connected!')"
```

## 8. Run Development Server (1 minute)

```bash
# Start the server
uvicorn app.main:app --reload

# Server will start at http://localhost:8000
```

## 9. Test API (2 minutes)

Open your browser:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000

## 10. Test with Postman/Insomnia (Optional)

### Register User
```http
POST http://localhost:8000/api/v1/users/register
Content-Type: application/json

{
  "username": "testuser",
  "email": "test@example.com",
  "password": "SecurePass123!",
  "full_name": "Test User"
}
```

### Login
```http
POST http://localhost:8000/api/v1/users/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "SecurePass123!"
}
```

## Common Issues

### Issue: "Module not found"
**Solution**: Make sure virtual environment is activated and dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "Supabase connection failed"
**Solution**: Check your `.env` file has correct Supabase credentials

### Issue: "Database tables not found"
**Solution**: Run the SQL schema creation in Supabase Dashboard

### Issue: "Port 8000 already in use"
**Solution**: Use a different port
```bash
uvicorn app.main:app --reload --port 8001
```

## Next Steps

After completing the quick start:

1. **Read** `SUPABASE_INTEGRATION.md` for detailed Supabase setup
2. **Follow** `NEXT_STEPS.md` for implementation roadmap
3. **Create** Pydantic schemas in `app/schemas/`
4. **Update** services to use Supabase client
5. **Implement** API endpoints with authentication
6. **Write** tests for your code
7. **Deploy** to production

## Useful Commands

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run on specific port
uvicorn app.main:app --reload --port 8001

# Run with Docker
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop Docker services
docker-compose down

# Run tests
pytest

# Format code
black app/ tests/

# Lint code
flake8 app/ tests/
```

## Project Structure

```
gsl-backend/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Core infrastructure
│   │   ├── supabase_client.py  # ⭐ Supabase integration
│   │   ├── database.py
│   │   ├── security.py
│   │   └── middleware.py
│   ├── services/        # Business logic
│   ├── ai/              # AI model integration
│   ├── utils/           # Utilities
│   └── main.py          # FastAPI app
├── tests/               # Test suite
├── .env                 # Environment variables (create this)
├── .env.example         # Example environment file
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Docker configuration
├── SUPABASE_INTEGRATION.md  # ⭐ Detailed Supabase guide
├── NEXT_STEPS.md        # ⭐ Implementation roadmap
└── README.md            # Project documentation
```

## Support

- **Documentation**: See `SUPABASE_INTEGRATION.md` and `NEXT_STEPS.md`
- **API Docs**: http://localhost:8000/docs (when server is running)
- **Supabase Docs**: https://supabase.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

**Total Setup Time**: ~45 minutes
**Status**: Ready to develop! 🚀