from fastapi import APIRouter, HTTPException, UploadFile, File, status
import os
from datetime import datetime
import shutil
import json
import time
from pdf_processor import ResumeProcessor
from image_processor import ImageProcessor

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def cleanup_temp_files(max_age_hours: int = 24):
    """Clean up old files in the uploads directory"""
    if not os.path.exists(UPLOAD_DIR):
        return
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Removed old temp file: {filename}")
                except Exception as e:
                    print(f"Failed to remove {filename}: {e}")

@router.post("/upload-test")
async def upload_file_test(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, DOC, DOCX files are allowed")
    
    # Create unique filename without user info for testing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Step 1: Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Step 2: Process resume with OpenAI (you need to set OPENAI_API_KEY environment variable)
        processor = ResumeProcessor()
        start_time = time.time()
        
        try:
            processing_result = processor.process_resume(file_path)
            processing_time = time.time() - start_time
            
            return {
                "message": "Resume uploaded and parsed successfully",
                "filename": filename,
                "original_filename": file.filename,
                "size": file.size,
                "upload_time": datetime.now().isoformat(),
                "processing_time": processing_time,
                "resume_data": processing_result.get("parsed_content", {}),
                "text_length": processing_result.get("text_length", 0),
                "parsing_status": processing_result.get("processing_status", "unknown")
            }
            
        except Exception as parsing_error:
            return {
                "message": "Resume uploaded but parsing failed",
                "filename": filename,
                "original_filename": file.filename,
                "size": file.size,
                "upload_time": datetime.now().isoformat(),
                "parsing_error": str(parsing_error),
                "parsing_status": "error"
            }
    
    except Exception as e:
        # Clean up file if upload fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")

@router.get("/files")
async def list_files():
    files = []
    
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        file_stat = os.stat(file_path)
        
        files.append({
            "filename": filename,
            "original_name": filename.split("_", 2)[2] if len(filename.split("_")) >= 3 else filename,
            "size": file_stat.st_size,
            "upload_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat()
        })
    
    return {"files": files}

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image with OpenAI
        processor = ImageProcessor()
        start_time = time.time()
        
        try:
            processing_result = processor.extract_text_from_image_sync(file_path)
            processing_time = time.time() - start_time
            
            return {
                "message": "Image uploaded and processed successfully",
                "filename": filename,
                "original_filename": file.filename,
                "size": file.size,
                "upload_time": datetime.now().isoformat(),
                "processing_time": processing_time,
                "extracted_text": processing_result.get("extracted_text", ""),
                "token_usage": processing_result.get("token_usage", 0),
                "processing_status": processing_result.get("status", "unknown")
            }
            
        except Exception as parsing_error:
            return {
                "message": "Image uploaded but text extraction failed",
                "filename": filename,
                "original_filename": file.filename,
                "size": file.size,
                "upload_time": datetime.now().isoformat(),
                "parsing_error": str(parsing_error),
                "processing_status": "error"
            }
    
    except Exception as e:
        # Clean up file if upload fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")

@router.post("/process-scorecards")
async def process_scorecards():
    """Process all scorecard images in the Scorecards folder"""
    scorecards_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Scorecards")
    
    if not os.path.exists(scorecards_path):
        raise HTTPException(status_code=404, detail="Scorecards folder not found")
    
    try:
        processor = ImageProcessor()
        start_time = time.time()
        
        result = processor.process_scorecard_folder(scorecards_path)
        processing_time = time.time() - start_time
        
        result["total_processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing scorecards: {str(e)}")

@router.get("/download-csv/{filename}")
async def download_csv(filename: str):
    """Download the generated CSV file"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )

@router.post("/debug-upload")
async def debug_upload(file: UploadFile = File(...)):
    """Debug endpoint to see exactly what's being returned"""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with debug output
        processor = ImageProcessor()
        result = processor.extract_text_from_image_sync(file_path)
        
        print("=== DEBUG ENDPOINT RESULT ===")
        print(f"Status: {result['status']}")
        print(f"Extracted text type: {type(result['extracted_text'])}")
        print(f"First 200 chars: {result['extracted_text'][:200]}")
        
        return {
            "debug_info": {
                "text_type": str(type(result['extracted_text'])),
                "text_length": len(result['extracted_text']),
                "starts_with": result['extracted_text'][:50],
                "is_json_parseable": "unknown"
            },
            "full_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
