import _path_resolver
import config

import json
import pathlib
import re
import time

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def resize_image(img, max_size=1024):
    """Resize image to speed up processing"""
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def clean_json_response(text):
    """Clean AI response and extract JSON"""
    # Remove markdown formatting
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON part
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

def natural_sort_key(path):
    """Natural sort: 1, 2, 3, ..., 10, 11 instead of 1, 10, 100, 11, 2"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path.name)]

def analyze_image(image_path):
    """Analyze a single image"""
    try:
        # Open and resize image
        img = Image.open(image_path)
        img = resize_image(img)
        
        # Correct JSON format (remove extra comma)
        prompt = """Analyze this product image and return ONLY a JSON object (no markdown, no explanation):
                  {
                    "name": "Product Name",
                    "category": "Product Category"
                  }

                  Return ONLY the JSON, nothing else."""
        response = model.generate_content([prompt, img])
        
        cleaned_text = clean_json_response(response.text)
        result = json.loads(cleaned_text)
        result["image_file"] = image_path.name
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        print(f"  Error: {e}")
        return {
            "image_file": image_path.name,
            "status": "error",
            "error": str(e)
        }

def bulk_analyze_and_save(save_path):
    image_folder = config.GLAMI_DATA_DIR
    folder = pathlib.Path(image_folder)
    
    # Get all image files, sort by name, take first 100
    img_paths = sorted(
        [
            file for file in folder.glob('*') 
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
        ],
        key=natural_sort_key
    )
    
    print(f"Found {len(img_paths)} images")
    bulk_size = 10
    start = 0
    end = bulk_size
    results = []
    
    for i in range(0, 20):
      if i > 0: 
          # Avoid exceeding API rate limit (15 requests per minute for free tier)
          time.sleep(60)
      for i, img_path in enumerate(img_paths[start:end], 1):
          print(f"[{i}/start:{start}, end:{end}] Analyzing {img_path.name}...")
          
          result = analyze_image(img_path)
          results.append(result)
          
          if result["status"] == "success":
              print(f"{result.get('category', 'N/A')} - {result.get('name', 'N/A')}")
          else:
              print(f"Failed: {result.get('error', 'Unknown error')}")
      start = end
      end += bulk_size
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {save_path}")


def restructure_product_json(file_path, modified_file_path):
    img_host_url = "pub-6cf2f88db8f14219bf79c4d284c2c63e.r2.dev"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modified_data = []
    id = 0
    for item in data:
      image_file = item["image_file"]
      image_url = f"https://{img_host_url}/{image_file}"
      
      item["image_url"] = image_url
      item["id"] = id
      id += 1
      
      modified_data.append(item)
    with open(modified_file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)
    
    print(f"Modifed results saved to {modified_file_path}")

if __name__ == "__main__":
    analyzed_save_path = "data/results.json"
    bulk_analyze_and_save(analyzed_save_path)
    restructure_product_json(analyzed_save_path, "data/products.json")