from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from upload import router as upload_router, cleanup_temp_files

app = FastAPI(title="Image Parser API", version="1.0.0")

# Clean up old temp files on startup
# @app.on_event("startup")
async def startup_event():
    cleanup_temp_files(max_age_hours=24)
    print("Cleaned up old temporary files on startup")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Alternative local port
        "https://frontend-6c9f9n89g-codeandteawithsamis-projects.vercel.app",  # Vercel frontend
        "https://*.vercel.app",  # All Vercel apps
        "*"  # Fallback for all origins
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


app.include_router(upload_router, tags=["upload"])

@app.get("/")
async def root():
    return {"message": "Image Parser API is running"}
