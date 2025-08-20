import os
from typing import Optional, Dict, Any
from PyPDF2 import PdfReader
from docx import Document
import openai
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeProcessor:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the resume processor with OpenAI client."""
        try:
            if openai_api_key:
                openai.api_key = openai_api_key
            else:
                # Try to get from environment variable
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                openai.api_key = api_key
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise Exception(f"OpenAI initialization failed: {str(e)}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text content from a DOCX file."""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text content from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {str(e)}")
                raise Exception(f"Failed to read TXT file: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            raise Exception(f"Failed to extract text from TXT: {str(e)}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file types."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_extension in ['.txt', '.doc']:  # .doc treated as txt for simplicity
            return self.extract_text_from_txt(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
    
    def parse_resume_with_llm(self, text: str, parsing_instruction: str = None) -> Dict[str, Any]:
        """Parse resume text using OpenAI GPT."""
        try:
            # Default resume parsing instruction
            if not parsing_instruction:
                parsing_instruction = """
                Please analyze this resume and extract the following information in JSON format:
                
                {
                    "personal_info": {
                        "name": "Full name",
                        "email": "Email address",
                        "phone": "Phone number",
                        "location": "City, State/Country",
                        "linkedin": "LinkedIn URL",
                        "portfolio": "Portfolio/Website URL"
                    },
                    "professional_summary": "Brief professional summary or objective",
                    "work_experience": [
                        {
                            "company": "Company name",
                            "position": "Job title",
                            "duration": "Start date - End date",
                            "location": "Job location",
                            "responsibilities": ["responsibility1", "responsibility2", "responsibility3"]
                        }
                    ],
                    "education": [
                        {
                            "institution": "School/University name",
                            "degree": "Degree type and major",
                            "graduation_date": "Graduation date",
                            "gpa": "GPA if mentioned",
                            "location": "School location"
                        }
                    ],
                    "skills": {
                        "technical_skills": ["skill1", "skill2", "skill3"],
                        "programming_languages": ["language1", "language2"],
                        "frameworks_tools": ["tool1", "tool2"],
                        "soft_skills": ["skill1", "skill2"]
                    },
                    "certifications": [
                        {
                            "name": "Certification name",
                            "issuer": "Issuing organization",
                            "date": "Date obtained",
                            "expiry": "Expiry date if applicable"
                        }
                    ],
                    "projects": [
                        {
                            "name": "Project name",
                            "description": "Brief description",
                            "technologies": ["tech1", "tech2"],
                            "duration": "Project duration",
                            "url": "Project URL if available"
                        }
                    ],
                    "languages": ["language1", "language2"],
                    "awards_achievements": ["achievement1", "achievement2"],
                    "years_of_experience": "Total years of experience",
                    "seniority_level": "Junior/Mid/Senior/Lead/Executive",
                    "key_domains": ["domain1", "domain2", "domain3"]
                }
                
                If any section is not present in the resume, use null or empty array as appropriate.
                Extract only the information that is explicitly mentioned in the resume.
                """
            
            # Truncate text if it's too long (OpenAI has token limits)
            max_chars = 12000  # Roughly 3000-4000 tokens
            if len(text) > max_chars:
                text = text[:max_chars] + "...[text truncated]"
                logger.warning("Text was truncated due to length limitations")
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional resume parser that extracts structured information from resumes. Always respond with valid JSON format."},
                    {"role": "user", "content": f"{parsing_instruction}\n\nResume text:\n{text}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            parsed_content = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to structured error if it fails
            try:
                return json.loads(parsed_content)
            except json.JSONDecodeError:
                logger.warning("LLM response was not valid JSON, returning structured error")
                return {
                    "personal_info": {"name": "Parse Error", "email": None, "phone": None, "location": None, "linkedin": None, "portfolio": None},
                    "professional_summary": f"Parsing failed - Raw response: {parsed_content}",
                    "work_experience": [],
                    "education": [],
                    "skills": {"technical_skills": [], "programming_languages": [], "frameworks_tools": [], "soft_skills": []},
                    "certifications": [],
                    "projects": [],
                    "languages": [],
                    "awards_achievements": [],
                    "years_of_experience": "Unknown",
                    "seniority_level": "Unknown",
                    "key_domains": [],
                    "parsing_error": "Invalid JSON response from AI model"
                }
                
        except Exception as e:
            logger.error(f"Error parsing document with LLM: {str(e)}")
            raise Exception(f"Failed to parse document with LLM: {str(e)}")
    
    def process_resume(self, file_path: str, parsing_instruction: str = None) -> Dict[str, Any]:
        """Complete resume processing pipeline: extract text and parse with LLM."""
        try:
            # Step 1: Extract text from resume
            logger.info(f"Extracting text from resume: {file_path}")
            extracted_text = self.extract_text_from_file(file_path)
            
            if not extracted_text or len(extracted_text.strip()) == 0:
                raise Exception("No text content found in the resume")
            
            # Step 2: Parse resume with LLM
            logger.info("Parsing resume with OpenAI GPT")
            parsed_result = self.parse_resume_with_llm(extracted_text, parsing_instruction)
            
            # Step 3: Combine results
            result = {
                "extracted_text": extracted_text,
                "parsed_content": parsed_result,
                "text_length": len(extracted_text),
                "processing_status": "success"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing resume {file_path}: {str(e)}")
            return {
                "extracted_text": "",
                "parsed_content": {
                    "personal_info": {"name": "Error", "email": None, "phone": None, "location": None, "linkedin": None, "portfolio": None},
                    "professional_summary": f"Processing failed: {str(e)}",
                    "work_experience": [],
                    "education": [],
                    "skills": {"technical_skills": [], "programming_languages": [], "frameworks_tools": [], "soft_skills": []},
                    "certifications": [],
                    "projects": [],
                    "languages": [],
                    "awards_achievements": [],
                    "years_of_experience": "Unknown",
                    "seniority_level": "Unknown",
                    "key_domains": [],
                    "processing_error": str(e)
                },
                "text_length": 0,
                "processing_status": "error",
                "error_message": str(e)
            }