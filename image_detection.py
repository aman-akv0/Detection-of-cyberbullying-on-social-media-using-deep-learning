import easyocr
from PIL import Image

# Initialize EasyOCR with CPU-optimized settings
reader = easyocr.Reader(
    lang_list=['en'],
    gpu=False,
    recog_network='english_g2',
    detector=True,
    model_storage_directory='models',
    download_enabled=True,
    verbose=False
)

def process_image(image_path):
    """Process image using EasyOCR with error handling"""
    try:
        # Basic image validation
        with Image.open(image_path) as img:
            img.verify()
            
        # Perform OCR
        results = reader.readtext(image_path, detail=0)
        return {'extracted_text': ' '.join(results).strip()}
    
    except Exception as e:
        return {'error': f"Image processing failed: {str(e)}"}

def preprocess_image(image_path):
    """Optimize image for OCR"""
    try:
        with Image.open(image_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((800, 600), Image.LANCZOS)  # Resize with anti-aliasing
            img.save(image_path, optimize=True, quality=85)
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")