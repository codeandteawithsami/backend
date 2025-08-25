#!/usr/bin/env python3
"""
Test script for the three images in the data folder
"""

import os
import sys
import asyncio
from image_processor import ImageProcessor
import json

def test_image(image_path, image_name):
    """Test a single image and print results"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_name}")
    print(f"Path: {image_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}")
        return
    
    processor = ImageProcessor()
    
    try:
        result = processor.extract_text_from_image_sync(image_path)
        
        print(f"Status: {result['status']}")
        print(f"Processing time: {result.get('processing_time', 'N/A')}")
        print(f"Token usage: {result.get('token_usage', 0)}")
        
        if result['status'] == 'success':
            extracted_text = result.get('extracted_text', '')
            if extracted_text:
                try:
                    # Try to parse and pretty print JSON
                    parsed_json = json.loads(extracted_text)
                    print("\nExtracted JSON:")
                    print(json.dumps(parsed_json, indent=2)[:2000] + "..." if len(extracted_text) > 2000 else json.dumps(parsed_json, indent=2))
                    
                    # Print key information
                    if 'courseInfo' in parsed_json and parsed_json['courseInfo']:
                        course_name = parsed_json['courseInfo'].get('name', 'Unknown')
                        print(f"\nCourse: {course_name}")
                    
                    if 'players' in parsed_json and parsed_json['players']:
                        print(f"Players found: {len(parsed_json['players'])}")
                        for i, player in enumerate(parsed_json['players'][:3]):  # Show first 3 players
                            name = player.get('name', f'Player {i+1}')
                            scores_count = len(player.get('scores', []))
                            print(f"  - {name} ({scores_count} holes)")
                    
                    if 'holes' in parsed_json and parsed_json['holes']:
                        print(f"Holes found: {len(parsed_json['holes'])}")
                        
                except json.JSONDecodeError as e:
                    print(f"WARNING: Invalid JSON returned")
                    print(f"JSON Error: {e}")
                    print(f"Raw text (first 500 chars): {extracted_text[:500]}")
            else:
                print("No text extracted")
        else:
            print(f"ERROR: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Test all three images in the data folder"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    print(f"Looking for images in: {data_dir}")
    
    # Define the three images to test
    test_images = [
        ('Scan2025-08-20_090428_003.jpg', 'Tournament Scorecard'),
        ('Scorecard Scan 1_000 (1).png', 'Multi-Player Scorecard'),
        ('Scorecard Scan 6_002.png', 'Course Yardage Chart')
    ]
    
    for filename, description in test_images:
        image_path = os.path.join(data_dir, filename)
        test_image(image_path, f"{description} ({filename})")
    
    print(f"\n{'='*60}")
    print("Testing completed!")
    print("If you see JSON parsing errors, the AI prompt may need further adjustment.")
    print("If you see extraction errors, check your OpenAI API key and connection.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()