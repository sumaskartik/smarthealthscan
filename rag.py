import base64
import json
import re
import logging
import os
import mimetypes
from typing import Dict, Any, List
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
from openai import OpenAI
import dotenv
from db import master_collection

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = OpenAI(api_key=openai_api_key)

def preprocess_challenging_documents(image_path: str) -> str:
    """Specialized preprocessing for challenging documents like government IDs"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # CRITICAL: Much more aggressive scaling for low-quality scans
        h, w = img.shape[:2]
        if w < 3000:  # Scale to 3000px minimum for government IDs
            scale_factor = 3000 / w
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"📏 Aggressive scaling: {w}x{h} → {new_w}x{new_h}")
        
        # Convert to RGB for PIL processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Step 1: Extreme noise reduction
        pil_img = pil_img.filter(ImageFilter.MedianFilter(size=7))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.0))
        
        # Step 2: Convert to grayscale
        gray = pil_img.convert('L')
        
        # Step 3: Extreme contrast enhancement
        contrast = ImageEnhance.Contrast(gray)
        high_contrast = contrast.enhance(4.0)  # Very high contrast
        
        # Step 4: Brightness adjustment
        brightness = ImageEnhance.Brightness(high_contrast)
        brightened = brightness.enhance(1.4)
        
        # Step 5: Extreme sharpening
        sharpness = ImageEnhance.Sharpness(brightened)
        ultra_sharp = sharpness.enhance(3.0)
        
        # Step 6: Convert back to OpenCV for advanced processing
        cv_img = cv2.cvtColor(np.array(ultra_sharp), cv2.COLOR_RGB2BGR)
        gray_cv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Step 7: Advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray_cv, h=30, templateWindowSize=7, searchWindowSize=21)
        
        # Step 8: Multiple thresholding approaches
        # Otsu thresholding
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
        
        # Combine both thresholding results
        combined = cv2.bitwise_and(otsu, adaptive)
        
        # Step 9: Morphological operations to connect broken text
        # Horizontal connection
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        h_morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, h_kernel)
        
        # Vertical connection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        v_morph = cv2.morphologyEx(h_morph, cv2.MORPH_CLOSE, v_kernel)
        
        # Clean small noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(v_morph, cv2.MORPH_OPEN, clean_kernel)
        
        # Step 10: Final dilation to make text thicker
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final_img = cv2.dilate(cleaned, dilate_kernel, iterations=1)
        
        temp_path = "temp_challenging_doc_processed.png"
        cv2.imwrite(temp_path, final_img)
        print(f"🎨 Challenging document preprocessing completed")
        return temp_path
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return image_path

def extract_text_challenging_documents(image_path: str) -> str:
    """Extract text from challenging documents with specialized OCR"""
    try:
        processed_path = preprocess_challenging_documents(image_path)
        
        # Specialized OCR configurations for government documents
        configs = [
            # Config 1: English + Hindi with strict character whitelist
            '--oem 3 --psm 6 -l eng+hin -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-+ ()',
            
            # Config 2: English only with character whitelist
            '--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-+ ()',
            
            # Config 3: Single text line mode
            '--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-+ ()',
            
            # Config 4: Treat image as single word
            '--oem 3 --psm 8 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-+ ()',
            
            # Config 5: Legacy engine
            '--oem 1 --psm 6 -l eng -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-+ ()',
            
            # Config 6: No character restrictions (fallback)
            '--oem 3 --psm 6 -l eng',
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            try:
                print(f"🔍 Challenging OCR Config {i+1}: {config[:50]}...")
                
                text = pytesseract.image_to_string(
                    Image.open(processed_path),
                    config=config,
                    timeout=30
                )
                
                # Advanced scoring for government documents
                # Look for government ID patterns
                gov_patterns = [
                    r'Government of India',
                    r'\d{4}\s*\d{4}\s*\d{4}',  # 12-digit numbers
                    r'\d{16}',  # 16-digit VID
                    r'DOB[:\s]*\d{2}/\d{2}/\d{4}',
                    r'MALE|FEMALE',
                    r'[A-Z]{2,}\s+[A-Z]{2,}',  # Names in caps
                ]
                
                pattern_score = sum(10 for pattern in gov_patterns 
                                  if re.search(pattern, text, re.IGNORECASE))
                
                # Text quality score
                readable_words = len([w for w in text.split() 
                                    if len(w) > 2 and w.isalpha()])
                
                numbers = len(re.findall(r'\d+', text))
                
                # Composite score
                total_score = pattern_score + readable_words + (numbers * 2)
                
                results.append({
                    'text': text,
                    'score': total_score,
                    'patterns': pattern_score,
                    'words': readable_words,
                    'numbers': numbers
                })
                
                print(f"   Score: {total_score} (patterns: {pattern_score}, words: {readable_words}, numbers: {numbers})")
                
            except Exception as e:
                print(f"   ❌ Config {i+1} failed: {e}")
                continue
        
        # Select best result
        if results:
            best = max(results, key=lambda x: x['score'])
            print(f"🏆 Best challenging document OCR score: {best['score']}")
            best_text = best['text']
        else:
            print("⚠️ All configs failed, using basic OCR")
            best_text = pytesseract.image_to_string(Image.open(processed_path))
        
        # Cleanup
        if os.path.exists(processed_path) and processed_path != image_path:
            os.remove(processed_path)
        
        # Post-process text
        if best_text:
            # Clean up common OCR errors
            cleaned = re.sub(r'[^\w\s.,:/()-]', ' ', best_text)  # Remove special chars
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
            lines = [line.strip() for line in cleaned.split('\n') if len(line.strip()) > 2]
            final_text = '\n'.join(lines)
            
            return final_text if len(final_text) > 30 else "Limited text detected"
        
        return "No text detected"
        
    except Exception as e:
        logging.error(f"❌ Challenging document OCR failed: {e}")
        return ""

# Replace your existing function with this enhanced version
def extract_from_image_with_openai(
    file_path: str, schema_fields: List[Dict]
) -> Dict[str, Any]:
    """Enhanced extraction for challenging documents"""
    
    field_keys = [f["key"] for f in schema_fields]
    
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type not in ["image/png", "image/jpeg", "image/webp"]:
            raise ValueError(f"❌ Unsupported format: {mime_type}")

        # Use challenging document OCR
        ocr_text = extract_text_challenging_documents(file_path)
        print(f"📄 Challenging Document OCR:\n{ocr_text}\n" + "="*60)
        
        if not ocr_text or len(ocr_text.strip()) < 30:
            logging.warning("⚠️ Very limited text extraction")
            return {"extracted": [], "missing": schema_fields}
        
        # Enhanced prompt for challenging documents
        enhanced_prompt = f"""
You are an expert at extracting information from challenging government documents and IDs.

DOCUMENT TEXT (may contain OCR errors):
{ocr_text}

FIELDS TO EXTRACT: {field_keys}

EXTRACTION INSTRUCTIONS:
1. This text is from a government identity document with possible OCR errors
2. Look for patterns like names, dates, ID numbers, addresses
3. Names are often in ALL CAPS
4. Dates may be in DD/MM/YYYY format
5. ID numbers may be 12 or 16 digits
6. Extract what you can recognize, even if text has errors
7. For unclear text, extract the most likely interpretation
8. Set null only if absolutely no information is found

Return JSON with extracted data:
"""

        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at extracting information from challenging government documents, even when OCR quality is poor. You can interpret garbled text and extract meaningful information."
                },
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.3,  # Slightly higher for interpretation
            max_tokens=2000,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content
        print(f"🤖 Enhanced AI Response:\n{raw_content}\n" + "="*60)

        try:
            parsed = json.loads(raw_content)
        except:
            parsed = {}

        # Map results
        extracted, missing = [], []
        for field in schema_fields:
            field_copy = field.copy()
            value = parsed.get(field["key"])
            
            if value and str(value).strip() not in ["", "null", "None"]:
                field_copy["value"] = [value] if not isinstance(value, list) else value
                extracted.append(field_copy)
                print(f"✅ {field['key']}: {value}")
            else:
                field_copy["value"] = []
                missing.append(field_copy)

        return {"extracted": extracted, "missing": missing}

    except Exception as e:
        logging.error(f"❌ Enhanced extraction error: {e}")
        return {"extracted": [], "missing": schema_fields}

# Keep your existing helper functions unchanged


# Keep existing helper functions
def get_document_schema(doc_type: str):
    try:
        schema_doc = master_collection.find_one(
            {"doc_type": doc_type}, {"_id": 0, "fields": 1}
        )
        if schema_doc and "fields" in schema_doc:
            return schema_doc["fields"]
        return None
    except Exception as e:
        return None

# Main execution
if __name__ == "__main__":
    # Works with any document schema
    schema_fields = get_document_schema("kyc_document")  # or "lab_report", "invoice", etc.
    image_path = "aadhaar_sample.png"
    
    result = extract_from_image_with_openai(image_path, schema_fields)
    
    print("📊 UNIVERSAL EXTRACTION RESULTS:")
    print(f"✅ Fields extracted: {len(result['extracted'])}")
    print(f"❌ Fields missing: {len(result['missing'])}")
