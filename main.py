from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from upload import router as upload_router, cleanup_temp_files

app = FastAPI(title="Image Parser API", version="1.0.0")

# Clean up old temp files on startup
@app.on_event("startup")
async def startup_event():
    cleanup_temp_files(max_age_hours=24)
    print("Cleaned up old temporary files on startup")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000","https://frontend-oqew8uoyr-codeandteawithsamis-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, tags=["upload"])

@app.get("/")
async def root():
    return {"message": "Image Parser API is running"}
