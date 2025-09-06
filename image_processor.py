import os
import base64
from openai import OpenAI
from PIL import Image
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
from dotenv import load_dotenv
from utils import detect_data

load_dotenv()

class ImageProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = {}  # Simple in-memory cache
        self.executor = ThreadPoolExecutor(max_workers=3)  # For parallel processing
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string with caching and compression"""
        # Create cache key from file path and modification time
        stat_info = os.stat(image_path)
        cache_key = f"{image_path}_{stat_info.st_mtime}_{stat_info.st_size}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Compress image if it's too large for faster processing
        try:
            # First, try to detect and crop the main object using YOLO
            print("Detection Image using YOLO model...")
            detections = detect_data(image_path, output_dir="temp_outputs")
            if detections["crop_image"]:
                img_path_to_encode = detections["crop_image"]
            else:
                img_path_to_encode = image_path
            print(f"Image to encode: {img_path_to_encode}")
            with Image.open(img_path_to_encode) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if image is too large (maintain aspect ratio)
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to bytes with compression
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=85, optimize=True)
                img_bytes.seek(0)
                
                encoded = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                self.cache[cache_key] = encoded
                return encoded
                
        except Exception as e:
            # Fallback to original method if compression fails
            print(f"Image compression failed, using original: {e}")
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                self.cache[cache_key] = encoded
                return encoded
    
    async def extract_text_from_image_async(self, image_path: str) -> Dict[str, Any]:
        """Async version of extract_text_from_image for better performance"""
        try:
            # Check cache first
            stat_info = os.stat(image_path)
            cache_key = f"extract_{image_path}_{stat_info.st_mtime}_{stat_info.st_size}"
            
            if cache_key in self.cache:
                print(f"Using cached result for {image_path}")
                return self.cache[cache_key]
            
            print(f"Processing image: {image_path}")
            
            # Encode the image
            base64_image = self.encode_image(image_path)
            print(f"Image encoded, size: {len(base64_image)} characters")
            
            # For debugging, let's try the simpler synchronous approach first
            return self.extract_text_from_image_sync(image_path)
            
        except Exception as e:
            print(f"Error in async extraction: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "extracted_text": '{"error": "Async processing failed", "cardType": "error", "confidence": 0}',
                "html_tables": "",
                "token_usage": 0,
                "processing_time": datetime.now().isoformat(),
                "confidence": 0
            }
    
    def extract_text_from_image_sync(self, image_path: str) -> Dict[str, Any]:
        """Synchronous version for debugging"""
        try:
            print(f"Starting sync processing for: {image_path}")
            
            # Encode the image
            base64_image = self.encode_image(image_path)
            print(f"Image encoded successfully, length: {len(base64_image)}")
            
            # Test OpenAI connection
            print("Testing OpenAI API connection...")
            
            # First, get JSON data
            json_response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": """
                            You are a COMPREHENSIVE golf scorecard data extraction expert. Your mission is to extract EVERY SINGLE piece of visible data from this scorecard image with COMPLETE thoroughness.

                            COMPREHENSIVE EXTRACTION RULES - MISS NOTHING:
                            1. SYSTEMATIC SCAN: Read EXACTLY as it appears - LEFT to RIGHT, TOP to BOTTOM
                            2. EXTRACT EVERYTHING: HANDWRITTEN and PRINTED text, numbers, symbols, marks
                            3. SCORECARD TYPES: Handle player scorecards, course info cards, tournament cards, practice rounds
                            4. PLAYER DATA: Read ALL player names, signatures, handicaps - check EVERY row for names
                            5. SCORE DATA: Extract ALL hole numbers, par values, scores - scan EVERY cell, EVERY column
                            6. COURSE DATA: Include ALL yardage information for EVERY tee color, ratings, slopes
                            7. EVENT DATA: Capture tournament names, dates, times, competition details, flight info
                            8. **SCORE ROWS WITHOUT NAMES**: If ANY row contains score numbers but NO player name:
                               - MANDATORY: Still extract ALL the score data from that row
                               - Look at previous row - if it has a player name, use that name (continuation)
                               - If no previous player name exists, set player name to "NONE"
                               - NEVER skip rows with score data - scores are priority
                               - Document which rows had missing names
                            9. **LOCATION TRACKING**: Note the general area/position where key data was found
                            10. **COMPLETENESS CHECK**: Scan entire image - headers, footers, margins, corners

                            Return ONLY this JSON structure with NO markdown formatting:
                            {
                                "confidence": 85,
                                "cardType": "scorecard",
                                "courseInfo": {
                                    "name": "Course Name",
                                    "location": "City, State", 
                                    "tournament": "Tournament Name if any",
                                    "date": "Date if shown",
                                    "time": "Tee time if shown"
                                },
                                "holes": [
                                    {
                                        "holeNumber": 1,
                                        "par": 4,
                                        "yardage": {
                                            "black": 425,
                                            "blue": 390,
                                            "white": 350,
                                            "gold": 320,
                                            "red": 280
                                        },
                                        "handicap": {
                                            "men": 7,
                                            "women": 9
                                        }
                                    }
                                ],
                                "players": [
                                    {
                                        "name": "Exact Name as Written",
                                        "tee": "Tee Color",
                                        "handicap": 12,
                                        "scores": [
                                            {
                                                "hole": 1,
                                                "score": 4,
                                                "putts": 2
                                            }
                                        ],
                                        "totals": {
                                            "front9": 42,
                                            "back9": 44,
                                            "total": 86
                                        }
                                    }
                                ],
                                "teeBoxes": [
                                    {
                                        "name": "Black",
                                        "totalYardage": {
                                            "front9": 3592,
                                            "back9": 3450,
                                            "total": 7042
                                        }
                                    }
                                ],
                                "totals": {
                                    "par": {
                                        "front9": 36,
                                        "back9": 36,
                                        "total": 72
                                    }
                                },
                                "competitionInfo": {
                                    "round": "Round number",
                                    "flight": "Flight info",
                                    "division": "Men's/Women's Championship etc"
                                }
                            }

                            COMPREHENSIVE EXTRACTION INSTRUCTIONS:
                            - **NAMES**: Extract EXACTLY as written: "Berndt, Phil" not "Phil Berndt"
                            - **HANDWRITING**: Read carefully: distinguish 4 from 9, 6 from 5, 1 from 7, 8 from 3
                            - **YARDAGES**: Include ALL visible yardages for EVERY tee color (Black, Blue, White, Gold, Red, etc.)
                            - **SIGNATURES**: Capture marker signatures, player attestations, official marks
                            - **SCORES**: Extract hole-by-hole scores in exact order - front 9, back 9, totals
                            - **TOURNAMENT**: Include competition details, round numbers, flight info, divisions
                            - **LAYOUTS**: Handle 9-hole, 18-hole, executive courses, par-3 courses
                            - **RATINGS**: Extract course rating, slope, handicap indexes for all tees
                            - **DATES/TIMES**: Capture any date stamps, tee times, round information
                            - **ADDITIONAL DATA**: Weather conditions, notes, special markings, logos
                            - **ROW SCANNING**: Check every row for data - even if it looks empty
                            - **CELL COMPLETENESS**: Extract partial numbers, unclear text with best guess
                            - **LOCATION NOTES**: Mention if data comes from top/bottom/left/right of scorecard
                            - CONFIDENCE: Set confidence (0-100) based on image quality, handwriting legibility, and data completeness
                              * 90-100: Crystal clear image, all text easily readable, complete scorecard data
                              * 70-89: Good quality, most text clear, minor uncertainty in 1-2 elements
                              * 50-69: Fair quality, some unclear handwriting or partial data
                              * 30-49: Poor quality, significant uncertainty in multiple elements
                              * 10-29: Very poor quality, mostly guesswork
                              * 0-9: Unreadable or no scorecard detected
                            """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Perform COMPREHENSIVE extraction of this golf scorecard. Scan every inch of the image systematically. Extract ALL visible data including: player names, scores, hole information, yardages, course details, tournament info, signatures, dates, and any other text or numbers. Don't skip any rows - even those without player names. Include data location context where helpful."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            print(f"JSON API call successful, tokens used: {json_response.usage.total_tokens if hasattr(json_response, 'usage') and json_response.usage else 'unknown'}")
            
            # Extract content
            extracted_text = json_response.choices[0].message.content
            print(f"Extracted text length: {len(extracted_text) if extracted_text else 0}")
            print(f"First 100 chars of extracted text: {extracted_text[:100] if extracted_text else 'None'}")
            
            # For now, skip HTML generation to isolate the issue
            html_tables = "HTML generation skipped for debugging"
            total_tokens = json_response.usage.total_tokens if hasattr(json_response, 'usage') and json_response.usage else 0
            
            # Clean and validate JSON
            if extracted_text:
                # Clean the response of common formatting issues
                extracted_text = self._clean_json_response(extracted_text)
                
                try:
                    # Validate JSON and extract confidence
                    parsed_json = json.loads(extracted_text)
                    confidence = parsed_json.get('confidence', 0)
                    print(f"JSON validation successful, confidence: {confidence}%")
                except json.JSONDecodeError as je:
                    print(f"JSON validation failed: {je}")
                    print(f"Raw response: {extracted_text[:200]}...")
                    
                    # Try multiple cleaning strategies
                    cleaned_text = self._extract_and_fix_json(extracted_text)
                    if cleaned_text:
                        extracted_text = cleaned_text
                        try:
                            parsed_json = json.loads(cleaned_text)
                            confidence = parsed_json.get('confidence', 0)
                            print(f"Successfully fixed JSON, confidence: {confidence}%")
                        except:
                            confidence = 0
                            print("Successfully fixed JSON")
                    else:
                        print("Could not fix JSON, returning error")
                        return {
                            "status": "error",
                            "error_message": f"Invalid JSON returned from AI: {str(je)}",
                            "extracted_text": f'{{"error": "Failed to parse response", "raw_response": "{extracted_text[:100]}...", "confidence": 0}}',
                            "html_tables": "",
                            "token_usage": total_tokens,
                            "processing_time": datetime.now().isoformat(),
                            "confidence": 0
                        }
            
            # Extract confidence from parsed JSON for the response
            try:
                parsed_result = json.loads(extracted_text) if extracted_text else {}
                result_confidence = parsed_result.get('confidence', 0)
            except:
                result_confidence = 0
            
            result = {
                "status": "success",
                "extracted_text": extracted_text or '{"error": "No text extracted", "cardType": "error", "confidence": 0}',
                "html_tables": html_tables,
                "token_usage": total_tokens,
                "processing_time": datetime.now().isoformat(),
                "confidence": result_confidence
            }
            
            print(f"Processing completed successfully. Status: {result['status']}")
            return result
            
        except Exception as e:
            print(f"Error in sync extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error_message": str(e),
                "extracted_text": f'{{"error": "Processing failed", "message": "{str(e)}", "cardType": "error", "confidence": 0}}',
                "html_tables": "",
                "token_usage": 0,
                "processing_time": datetime.now().isoformat(),
                "confidence": 0
            }
    
    async def _get_json_response(self, base64_image: str):
        """Get JSON response from OpenAI API"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON data extractor. You MUST return ONLY valid JSON. No markdown, no ```json blocks, no explanations - just pure JSON starting with { and ending with }."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this golf scorecard and extract ALL data into JSON format. Return ONLY the JSON object - nothing else."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
        )
    
    async def _get_table_response(self, base64_image: str):
        """Get table response from OpenAI API"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(
                self.client.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HTML table generator. Create beautiful, well-structured HTML tables from golf scorecard data. Use Tailwind CSS classes for styling."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this golf scorecard and create 3 HTML tables with Tailwind CSS styling:

1. **Course Information Table**: Course name, tees, par, date/time, etc.
2. **Player Scores Table**: Players as rows, holes as columns, with individual scores and totals
3. **Hole Information Table**: Hole number, par, yardage, handicap, etc.

Requirements:
- Use Tailwind CSS classes for styling
- Include proper table structure with <thead> and <tbody>
- Add responsive classes like 'overflow-x-auto', 'table-auto'
- Use colors: bg-blue-50, bg-green-50, text-green-600, etc.
- Color-code scores: green for under par, red for over par, blue for par
- Add hover effects: hover:bg-gray-50
- Make tables clean and professional
- Use semantic headings like <h3> for table titles

Return ONLY the HTML content - no explanations, no markdown, no ```html blocks."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=3000
            )
        )
    
    def _process_responses(self, json_response, table_response) -> Dict[str, Any]:
        """Process API responses and return formatted result"""
            
        extracted_text = json_response.choices[0].message.content
        html_tables = table_response.choices[0].message.content
        total_tokens = (json_response.usage.total_tokens if hasattr(json_response, 'usage') and json_response.usage else 0) + \
                      (table_response.usage.total_tokens if hasattr(table_response, 'usage') and table_response.usage else 0)
        
        # Clean up the response in case it has markdown formatting
        if extracted_text.startswith('```'):
            lines = extracted_text.split('\n')
            start_idx = 1 if lines[0].startswith('```') else 0
            end_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == '```':
                    end_idx = i
                    break
            extracted_text = '\n'.join(lines[start_idx:end_idx])
        
        # Validate JSON quickly
        try:
            json.loads(extracted_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
            if json_match:
                extracted_text = json_match.group()
        
        return {
            "status": "success",
            "extracted_text": extracted_text,
            "html_tables": html_tables,
            "token_usage": total_tokens,
            "processing_time": datetime.now().isoformat()
        }
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Synchronous wrapper for async method to maintain compatibility"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.extract_text_from_image_async(image_path))
        finally:
            loop.close()
    
    async def process_multiple_images_async(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images concurrently for better performance"""
        tasks = []
        for image_path in image_paths:
            task = asyncio.create_task(self.extract_text_from_image_async(image_path))
            tasks.append((task, image_path))
        
        results = []
        for task, image_path in tasks:
            result = await task
            result["image_path"] = image_path
            result["image_name"] = os.path.basename(image_path)
            results.append(result)
        
        return results
    
    def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for backward compatibility"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.process_multiple_images_async(image_paths))
        finally:
            loop.close()
    
    def export_to_csv(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """Export extracted text results to CSV"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"extracted_text_{timestamp}.csv"
        
        # Create DataFrame from results
        df_data = []
        for result in results:
            df_data.append({
                "image_name": result.get("image_name", ""),
                "image_path": result.get("image_path", ""),
                "status": result.get("status", ""),
                "extracted_text": result.get("extracted_text", ""),
                "error_message": result.get("error_message", ""),
                "token_usage": result.get("token_usage", 0),
                "processing_time": result.get("processing_time", "")
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    def process_scorecard_folder(self, folder_path: str) -> Dict[str, Any]:
        """Process all images in the Scorecards folder"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        image_paths = []
        
        # Get all image files from the folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(folder_path, filename))
        
        if not image_paths:
            return {
                "status": "error",
                "message": "No image files found in the specified folder"
            }
        
        # Process all images
        results = self.process_multiple_images(image_paths)
        
        # Export to CSV
        csv_path = self.export_to_csv(results)
        
        # Calculate summary statistics
        successful_extractions = len([r for r in results if r["status"] == "success"])
        total_tokens = sum(r.get("token_usage", 0) for r in results)
        
        return {
            "status": "success",
            "total_images": len(image_paths),
            "successful_extractions": successful_extractions,
            "failed_extractions": len(image_paths) - successful_extractions,
            "total_tokens_used": total_tokens,
            "csv_output_path": csv_path,
            "results": results
        }
    
    def _clean_json_response(self, text: str) -> str:
        """Clean common JSON formatting issues from AI response"""
        if not text:
            return text
            
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
            
        if text.endswith('```'):
            text = text[:-3]
            
        # Remove any text before the first {
        start_idx = text.find('{')
        if start_idx > 0:
            text = text[start_idx:]
            
        # Remove any text after the last }
        end_idx = text.rfind('}') + 1
        if end_idx > 0:
            text = text[:end_idx]
            
        return text.strip()
    
    def _extract_and_fix_json(self, text: str) -> str:
        """Try multiple strategies to extract and fix JSON"""
        import re
        
        # Strategy 1: Find JSON object with regex
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
            try:
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed_text = re.sub(r',\s*([}\]])', r'\1', text)
            # Fix single quotes to double quotes
            fixed_text = re.sub(r"'", '"', fixed_text)
            # Fix missing quotes around keys
            fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
            
            # Try to fix incomplete JSON by ensuring it ends properly
            if fixed_text.count('{') > fixed_text.count('}'):
                # Add missing closing braces
                missing_braces = fixed_text.count('{') - fixed_text.count('}')
                fixed_text += '}' * missing_braces
            
            if fixed_text.count('[') > fixed_text.count(']'):
                # Add missing closing brackets
                missing_brackets = fixed_text.count('[') - fixed_text.count(']')
                fixed_text += ']' * missing_brackets
            
            json.loads(fixed_text)
            return fixed_text
        except json.JSONDecodeError:
            pass
            
        # Strategy 3: Return minimal valid JSON for errors
        try:
            # Don't truncate - keep full text for debugging
            escaped_text = text.replace('"', '\\"').replace('\n', '\\n')
            return f'{{"error": "Could not parse scorecard", "cardType": "error", "extractedText": "{escaped_text}"}}'
        except:
            return '{"error": "Complete parsing failure", "cardType": "error"}'
    
