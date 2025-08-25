import uvicorn
from main import app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,  # Use 1 worker for development, increase for production
        loop="asyncio",
        http="httptools",
        access_log=False,  # Disable access logs for better performance
        log_level="info"
    )