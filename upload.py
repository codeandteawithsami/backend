from fastapi import APIRouter, HTTPException, UploadFile, File, status
import os
from datetime import datetime, timedelta
import shutil
import json
import time
from pdf_processor import ResumeProcessor
from image_processor import ImageProcessor

router = APIRouter()

UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files(max_age_hours: int = 24):
    """Clean up temporary files older than max_age_hours"""
    try:
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=max_age_hours)
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time:
                    os.remove(file_path)
                    print(f"Cleaned up old temp file: {filename}")
    except Exception as e:
        print(f"Error during temp cleanup: {e}")

def get_temp_file_info():
    """Get information about files in temp directory"""
    files_info = []
    try:
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                stat_info = os.stat(file_path)
                files_info.append({
                    "filename": filename,
                    "size": stat_info.st_size,
                    "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                })
    except Exception as e:
        print(f"Error getting temp file info: {e}")
    
    return files_info

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
    file_path = os.path.join(TEMP_DIR, filename)
    
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
    
    # Create unique filename in temp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image with OpenAI
        processor = ImageProcessor()
        start_time = time.time()
        
        try:
            # Use sync processing to avoid event loop conflicts
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
    """Process all scorecard images in the Scorecards folder with async processing"""
    scorecards_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Scorecards")
    
    if not os.path.exists(scorecards_path):
        raise HTTPException(status_code=404, detail="Scorecards folder not found")
    
    try:
        processor = ImageProcessor()
        start_time = time.time()
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        image_paths = []
        
        for filename in os.listdir(scorecards_path):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(scorecards_path, filename))
        
        if not image_paths:
            raise HTTPException(status_code=404, detail="No image files found in the Scorecards folder")
        
        # Process all images using sync method to avoid event loop conflicts
        results = processor.process_multiple_images(image_paths)
        
        # Export to CSV
        csv_path = processor.export_to_csv(results)
        
        # Calculate summary statistics
        successful_extractions = len([r for r in results if r["status"] == "success"])
        total_tokens = sum(r.get("token_usage", 0) for r in results)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "total_images": len(image_paths),
            "successful_extractions": successful_extractions,
            "failed_extractions": len(image_paths) - successful_extractions,
            "total_tokens_used": total_tokens,
            "csv_output_path": csv_path,
            "total_processing_time": processing_time,
            "results": results
        }
        
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
    file_path = os.path.join(TEMP_DIR, filename)
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with debug output using sync method
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

@router.get("/temp-files")
async def list_temp_files():
    """List all files in the temp directory"""
    try:
        files_info = get_temp_file_info()
        return {
            "temp_directory": TEMP_DIR,
            "total_files": len(files_info),
            "files": files_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing temp files: {str(e)}")

@router.post("/cleanup-temp")
async def cleanup_temp():
    """Clean up old temporary files (older than 24 hours)"""
    try:
        files_before = len(get_temp_file_info())
        cleanup_temp_files(max_age_hours=24)
        files_after = len(get_temp_file_info())
        
        return {
            "message": "Temp cleanup completed",
            "files_before": files_before,
            "files_after": files_after,
            "files_removed": files_before - files_after
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

@router.delete("/temp-files/clear-all")
async def clear_all_temp_files():
    """Clear all files from the temp directory"""
    try:
        files_info = get_temp_file_info()
        files_count = len(files_info)
        
        for file_info in files_info:
            file_path = os.path.join(TEMP_DIR, file_info["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return {
            "message": f"Cleared all temp files",
            "files_removed": files_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing temp files: {str(e)}")

@router.delete("/temp-files/{filename}")
async def delete_temp_file(filename: str):
    """Delete a specific file from the temp directory"""
    try:
        file_path = os.path.join(TEMP_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        os.remove(file_path)
        
        return {
            "message": f"File '{filename}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
