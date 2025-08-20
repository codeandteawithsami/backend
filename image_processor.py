import os
import base64
from openai import OpenAI
from PIL import Image
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List

class ImageProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image using GPT-4 Vision"""
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Create the prompt for text extraction using GPT-4o with JSON format
            response = self.client.chat.completions.create(
                model="gpt-5",
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
            extracted_text = response.choices[0].message.content
            token_usage = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            
            print(f"=== RAW GPT-4o RESPONSE ===")
            print(f"First 100 chars: {extracted_text[:100]}")
            print(f"Starts with backticks: {extracted_text.startswith('```')}")
            
            # Clean up the response in case it has markdown formatting
            if extracted_text.startswith('```'):
                # Remove markdown code blocks
                lines = extracted_text.split('\n')
                # Find the start and end of the actual content
                start_idx = 1 if lines[0].startswith('```') else 0
                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == '```':
                        end_idx = i
                        break
                extracted_text = '\n'.join(lines[start_idx:end_idx])
            
            # Validate that we got valid JSON
            try:
                import json
                json.loads(extracted_text)  # This will raise an exception if not valid JSON
                print("Valid JSON received from GPT-4o")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON from GPT-4o: {e}")
                print(f"Raw response: {extracted_text[:200]}...")
                # If still not valid JSON, try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                if json_match:
                    extracted_text = json_match.group()
                    try:
                        json.loads(extracted_text)
                        print("Extracted valid JSON from response")
                    except:
                        print("Could not extract valid JSON")
            
            return {
                "status": "success",
                "extracted_text": extracted_text,
                "token_usage": token_usage,
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "extracted_text": "",
                "token_usage": 0,
                "processing_time": datetime.now().isoformat()
            }
    
    def process_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple images and extract text from each"""
        results = []
        
        for image_path in image_paths:
            result = self.extract_text_from_image(image_path)
            result["image_path"] = image_path
            result["image_name"] = os.path.basename(image_path)
            results.append(result)
        
        return results
    
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