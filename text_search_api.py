"""
Text-based Product Search with Image Download
User can search by text query (e.g., "rolex watch") and get actual images
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import requests
from PIL import Image
from io import BytesIO
import zipfile
import re
from typing import List, Dict
import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(title="Text-based Product Search with Images")

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
DB_NAME = os.getenv("DATABASE_NAME", "watchvine_refined")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ============================================================================
# BRAND NAME MAPPING (User input → Database format)
# Website stores brand names with modified spelling for copyright reasons
# ============================================================================
BRAND_MAPPING = {
    # Watch brands
    'fossil': 'fossi_l',
    'tissot': 'tisso_t',
    'armani': 'arman_i',
    'tommy': 'tomm_y',
    'tommy hilfiger': 'tomm_y',
    'rolex': 'role_x',
    'rado': 'rad_o',
    'omega': 'omeg_a',
    'patek': 'Patek_Philippe',
    'patek philippe': 'Patek_Philippe',
    'patek phillips': 'Patek_Philippe',
    'hublot': 'hublo_t',
    'cartier': 'cartie_r',
    'ap': 'Audemars',
    'audemars': 'Audemars',
    'tag': 'tag',
    'tag heuer': 'tag',
    'tag huer': 'tag',
    'mk': 'mic',
    'michael kors': 'mic',
    'alix': 'alix',
    'naviforce': 'naviforce',
    'reward': 'reward',
    'ax': 'ax',
    'armani exchange': 'arman_i',
    
    # Add more as needed
}

def normalize_brand_name(keyword: str) -> str:
    """
    Convert user-friendly brand name to database format.
    Example: "armani" → "arman_i", "rolex" → "role_x"
    """
    keyword_lower = keyword.lower().strip()
    
    # Check exact match first
    if keyword_lower in BRAND_MAPPING:
        return BRAND_MAPPING[keyword_lower]
    
    # Check if keyword contains a brand name
    for user_name, db_name in BRAND_MAPPING.items():
        if user_name in keyword_lower or keyword_lower in user_name:
            return db_name
    
    # Return original if no mapping found
    return keyword


# Generic product type keywords that may not appear in product names
GENERIC_TYPES = ['watch', 'watches', 'shoe', 'shoes', 'bag', 'bags', 'sunglass', 'sunglasses']

def is_generic_type(keyword: str) -> bool:
    """Check if keyword is a generic product type"""
    return keyword.lower() in GENERIC_TYPES

# Temporary folder for images
TEMP_FOLDER = "temp_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)


class SearchRequest(BaseModel):
    query: str  # e.g., "rolex watch", "michael kors bag"
    max_results: int = 10


def download_image_from_url(url: str, save_path: str) -> bool:
    """Download image from URL and save locally."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def search_products_by_text(query: str, max_results: int = 10, category_filter: str = None) -> List[Dict]:
    """
    Search products in MongoDB with SMART keyword matching + Brand name normalization + Category filtering.
    ALL keywords must be present in product name.
    Handles: brand name mapping, underscores, plurals, word boundaries, category filtering
    
    Example: "rolex watch" → Converts "rolex" to "role_x" → Matches "role_x watch" in mens_watch category
    """
    # Split query into keywords
    keywords = [kw.strip().lower() for kw in query.strip().split() if len(kw.strip()) > 1]
    
    if not keywords:
        return []
    
    # STEP 1: Normalize brand names (rolex → role_x, armani → arman_i, etc.)
    normalized_keywords = []
    essential_keywords = []  # Keywords that MUST be in product name
    
    for keyword in keywords:
        normalized = normalize_brand_name(keyword)
        
        # Skip generic type keywords if no brand specified
        # e.g., "watch" alone is too generic, but "rolex watch" → keep "rolex", skip "watch"
        if is_generic_type(normalized) and len(keywords) > 1:
            print(f"⏭️ Skipping generic type: '{keyword}' (brand search more specific)")
            continue
        
        normalized_keywords.append(normalized)
        essential_keywords.append(normalized)
    
    # Log the transformation
    if normalized_keywords != keywords:
        print(f"🔄 Keywords: {keywords} → {normalized_keywords}")
    
    # Use only essential keywords for search
    if not essential_keywords:
        return []  # All keywords were generic, no specific search possible
    
    keywords = essential_keywords
    
    # Build smart patterns for ALL keywords
    patterns = []
    for keyword in keywords:
        # Handle common plurals (watch/watches, shoe/shoes, bag/bags)
        if keyword.endswith('s'):
            # If already plural, make 's' optional: watches? → watch or watches
            base = keyword[:-1]
            pattern = re.compile(f"{re.escape(base)}e?s?", re.IGNORECASE)
        elif keyword in ['watch', 'shoe', 'bag', 'glass', 'sunglass']:
            # Add optional 'es' or 's' for singular: watch → watch or watches
            pattern = re.compile(f"{re.escape(keyword)}e?s?", re.IGNORECASE)
        else:
            # Regular keyword - simple case-insensitive match
            # Brand names are already normalized (role_x, arman_i, etc.)
            # Just search for the exact normalized brand name
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        patterns.append(pattern)
    
    # ALL patterns MUST match
    and_conditions = [{"name": {"$regex": pattern}} for pattern in patterns]
    
    # Add category filter if provided
    if category_filter:
        and_conditions.append({"category_key": category_filter})
        print(f"📂 Category filter applied: {category_filter}")
    
    results = list(collection.find(
        {"$and": and_conditions},
        {"name": 1, "price": 1, "image_urls": 1, "url": 1, "category": 1, "category_key": 1}
    ).limit(max_results))
    
    print(f"🔍 SMART Search: '{query}'")
    print(f"📋 Keywords: {keywords}")
    if category_filter:
        print(f"📂 Category: {category_filter}")
    print(f"✅ Found {len(results)} products (ALL keywords matched)")
    
    return results


@app.get("/")
async def root():
    return {
        "service": "Text-based Product Search with Images",
        "endpoints": {
            "/search": "POST - Search products by text and get images",
            "/search/images": "POST - Search and download images as ZIP",
            "/products/count": "GET - Get total products in database"
        }
    }


@app.post("/search")
async def search_products(request: SearchRequest):
    """
    Search products by text query and return product info with image URLs.
    
    Example request:
    {
        "query": "rolex",
        "max_results": 10
    }
    """
    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    # Search in MongoDB
    products = search_products_by_text(request.query, request.max_results)
    
    if not products:
        return JSONResponse(
            status_code=404,
            content={
                "status": "no_match",
                "message": f"No products found matching '{request.query}'",
                "total_results": 0
            }
        )
    
    # Format results
    results = []
    total_images = 0
    
    for product in products:
        product_data = {
            "product_name": product.get("name", "Unknown"),
            "price": product.get("price", "N/A"),
            "product_url": product.get("url", ""),
            "images": product.get("image_urls", []),
            "image_count": len(product.get("image_urls", []))
        }
        results.append(product_data)
        total_images += product_data["image_count"]
    
    return {
        "status": "success",
        "query": request.query,
        "total_products": len(results),
        "total_images": total_images,
        "products": results
    }


@app.post("/search/download")
async def search_and_download_images(request: SearchRequest):
    """
    Search products and download all images as ZIP file.
    Returns a downloadable ZIP containing all product images.
    
    Example request:
    {
        "query": "rolex",
        "max_results": 5
    }
    """
    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    # Search products
    products = search_products_by_text(request.query, request.max_results)
    
    if not products:
        raise HTTPException(
            status_code=404, 
            detail=f"No products found matching '{request.query}'"
        )
    
    # Create ZIP file in memory
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        image_counter = 1
        
        for product_idx, product in enumerate(products, 1):
            product_name = product.get("name", "Unknown")
            # Sanitize product name for folder
            safe_name = re.sub(r'[^\w\s-]', '', product_name).strip().replace(' ', '_')[:50]
            price = product.get("price", "N/A")
            image_urls = product.get("image_urls", [])
            
            # Create product info text file
            info_content = f"Product: {product_name}\n"
            info_content += f"Price: ₹{price}\n"
            info_content += f"URL: {product.get('url', 'N/A')}\n"
            info_content += f"Images: {len(image_urls)}\n"
            
            zip_file.writestr(f"{safe_name}/product_info.txt", info_content)
            
            # Download and add images
            for img_idx, img_url in enumerate(image_urls, 1):
                try:
                    # Download image
                    temp_path = os.path.join(TEMP_FOLDER, f"temp_{image_counter}.jpg")
                    
                    if download_image_from_url(img_url, temp_path):
                        # Add to ZIP
                        zip_file.write(
                            temp_path, 
                            f"{safe_name}/image_{img_idx}.jpg"
                        )
                        # Cleanup
                        os.remove(temp_path)
                        image_counter += 1
                        
                except Exception as e:
                    print(f"Error processing image {img_url}: {e}")
                    continue
    
    # Prepare ZIP for download
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={request.query.replace(' ', '_')}_products.zip"
        }
    )


def download_and_convert_to_base64(image_url: str) -> str:
    """Download image and convert to base64 string (plain base64, no prefix)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Convert to base64 - NO PREFIX for Evolution API
            base64_string = base64.b64encode(response.content).decode('utf-8')
            return base64_string
        return None
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return None


def download_all_images_parallel(image_urls: List[str], max_workers: int = 10) -> List[str]:
    """Download multiple images in parallel and return base64 strings in order."""
    base64_images = [None] * len(image_urls)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all downloads and keep track of their index
        future_to_index = {
            executor.submit(download_and_convert_to_base64, url): idx 
            for idx, url in enumerate(image_urls)
        }
        
        # Collect results and place them in correct position
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                if result:
                    base64_images[idx] = result
            except Exception as e:
                print(f"Error in parallel download at index {idx}: {e}")
    
    # Filter out None values (failed downloads)
    return [img for img in base64_images if img is not None]


@app.post("/search/images-list")
async def search_and_get_images(request: SearchRequest):
    """
    Search products and return images as base64 (FAST - Parallel download).
    Optimized for WhatsApp bot integration with parallel image processing.
    """
    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    # Search products
    products = search_products_by_text(request.query, request.max_results)
    
    if not products:
        return {
            "status": "no_match",
            "message": f"No products found matching '{request.query}'",
            "products": []
        }
    
    results = []
    
    # Collect all image URLs for parallel download
    all_image_urls = []
    product_image_mapping = []  # Track which images belong to which product
    
    for product in products:
        product_name = product.get("name", "Unknown")
        price = product.get("price", "N/A")
        image_urls = product.get("image_urls", [])
        
        if image_urls:
            # Take all images (not just first one)
            product_image_mapping.append({
                "product_name": product_name,
                "price": price,
                "product_url": product.get("url", ""),
                "start_index": len(all_image_urls),
                "image_count": len(image_urls)
            })
            all_image_urls.extend(image_urls)
    
    # Download all images in parallel
    print(f"Downloading {len(all_image_urls)} images in parallel...")
    base64_images = download_all_images_parallel(all_image_urls, max_workers=15)
    
    # Map base64 images back to products
    for product_info in product_image_mapping:
        start_idx = product_info["start_index"]
        end_idx = start_idx + product_info["image_count"]
        
        # Get base64 images for this product
        product_images = []
        for i in range(start_idx, end_idx):
            if i < len(base64_images):
                product_images.append(base64_images[i])
        
        if product_images:
            results.append({
                "product_name": product_info["product_name"],
                "price": product_info["price"],
                "product_url": product_info["product_url"],
                "images_base64": product_images,
                "total_images": len(product_images)
            })
    
    return {
        "status": "success",
        "query": request.query,
        "total_products": len(results),
        "total_images": len(base64_images),
        "products": results
    }


@app.get("/products/count")
async def get_product_count():
    """Get total number of products in database."""
    count = collection.count_documents({})
    return {
        "total_products": count,
        "database": DB_NAME,
        "collection": COLLECTION_NAME
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test MongoDB connection
        collection.find_one()
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "database": db_status,
        "total_products": collection.count_documents({})
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🚀 Text-based Product Search API with Image Download")
    print("="*80)
    print("\nEndpoints:")
    print("  POST /search              - Search products by text")
    print("  POST /search/download     - Download images as ZIP")
    print("  POST /search/images-list  - Get images as base64 (for WhatsApp)")
    print("  GET  /products/count      - Get total products")
    print("  GET  /health              - Health check")
    print("\nStarting server on http://localhost:8001")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
