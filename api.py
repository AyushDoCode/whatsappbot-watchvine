"""
Visual Search Engine - FastAPI Server
Handles image-based product search using CLIP embeddings and FAISS.
"""

import os
import pickle
import re
import numpy as np
from io import BytesIO
from typing import Dict, Optional, List
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configuration
MODEL_NAME = "clip-ViT-B-32"
INDEX_FILE = "vector_index.bin"
METADATA_FILE = "metadata.pkl"
# Adjusted thresholds for large dataset (8000+ products)
SIMILARITY_THRESHOLD_HIGH = 0.75  # High confidence threshold (75%+) - LOWERED from 0.85
SIMILARITY_THRESHOLD_MEDIUM = 0.65  # Medium confidence threshold (65-75%) - LOWERED from 0.75
SIMILARITY_THRESHOLD_LOW = 0.55  # Low confidence threshold (55-65%) - LOWERED from 0.70
MAX_DISTANCE_THRESHOLD = 0.35  # L2 distance threshold (lower is better)
MAX_IMAGE_SIZE_MB = 10
TARGET_IMAGE_SIZE = (224, 224)  # Standard CLIP input size

# Product categories for filtering
PRODUCT_CATEGORIES = {
    'watch': ['watch', 'wrist', 'timepiece', 'chronograph'],
    'bag': ['bag', 'purse', 'handbag', 'tote', 'clutch', 'satchel', 'backpack', 'shoulder'],
    'sunglasses': ['sunglasses', 'sunglass', 'eyewear', 'shades', 'glasses'],
    'shoes': ['shoes', 'shoe', 'footwear', 'sneakers', 'boots', 'sandals'],
    'wallet': ['wallet', 'purse'],
    'bracelet': ['bracelet', 'bangle', 'jewellery', 'jewelry']
}

# Global variables for model and index
model = None
index = None
metadata = None

# Initialize FastAPI app
app = FastAPI(
    title="Visual Search Engine API",
    description="Search for watches using image similarity with CLIP + FAISS",
    version="1.0.0"
)


def load_resources():
    """Load CLIP model, FAISS index, and metadata on startup."""
    global model, index, metadata
    
    print("🚀 Loading resources...")
    
    # Load CLIP model
    print(f"🤖 Loading CLIP model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Model loaded")
    
    # Load FAISS index (optional - API can run without it)
    if not os.path.exists(INDEX_FILE):
        print(f"⚠️ FAISS index not found: {INDEX_FILE}")
        print("⚠️ API will run in limited mode. Run indexer.py to enable search functionality.")
        index = None
        metadata = None
        return
    
    print(f"📂 Loading FAISS index: {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)
    print(f"✅ Index loaded with {index.ntotal} vectors")
    
    # Load metadata
    if not os.path.exists(METADATA_FILE):
        print(f"⚠️ Metadata not found: {METADATA_FILE}")
        print("⚠️ Index found but metadata missing. API will run in limited mode.")
        index = None
        metadata = None
        return
    
    print(f"📂 Loading metadata: {METADATA_FILE}")
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✅ Metadata loaded with {len(metadata)} entries")


@app.on_event("startup")
async def startup_event():
    """Load resources when API starts."""
    try:
        load_resources()
        if index is not None and metadata is not None:
            print("🎉 API ready to serve requests!")
        else:
            print("⚠️ API started in limited mode (index not available)")
            print("   Run indexer.py to enable search functionality")
    except Exception as e:
        print(f"❌ Failed to load resources: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Visual Search Engine API",
        "status": "running",
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "metadata_loaded": metadata is not None,
        "total_vectors": index.ntotal if index else 0
    }


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Resources not loaded")
    
    # Count unique products
    unique_products = len(set(m['product_name'] for m in metadata))
    
    # Count products by category
    category_counts = {}
    for m in metadata:
        cat = detect_category_from_metadata(m.get('product_name', ''))
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        else:
            category_counts['uncategorized'] = category_counts.get('uncategorized', 0) + 1
    
    return {
        "total_vectors": index.ntotal,
        "total_images": len(metadata),
        "unique_products": unique_products,
        "embedding_dimension": index.d,
        "thresholds": {
            "high_confidence": f"{SIMILARITY_THRESHOLD_HIGH*100:.0f}% (≥{SIMILARITY_THRESHOLD_HIGH})",
            "medium_confidence": f"{SIMILARITY_THRESHOLD_MEDIUM*100:.0f}%-{SIMILARITY_THRESHOLD_HIGH*100:.0f}% ({SIMILARITY_THRESHOLD_MEDIUM}-{SIMILARITY_THRESHOLD_HIGH})",
            "low_confidence": f"{SIMILARITY_THRESHOLD_LOW*100:.0f}%-{SIMILARITY_THRESHOLD_MEDIUM*100:.0f}% ({SIMILARITY_THRESHOLD_LOW}-{SIMILARITY_THRESHOLD_MEDIUM})",
            "rejected": f"<{SIMILARITY_THRESHOLD_LOW*100:.0f}%"
        },
        "category_distribution": category_counts,
        "note": "Similarity score: 0-1 (higher is better), L2 distance: 0+ (lower is better)"
    }


def detect_category_from_metadata(product_name: str) -> Optional[str]:
    """Detect product category from product name."""
    product_name_lower = product_name.lower()
    
    for category, keywords in PRODUCT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in product_name_lower:
                return category
    
    return None


def get_category_indices(category: Optional[str]) -> List[int]:
    """Get indices of products that match the given category."""
    if category is None or metadata is None:
        return list(range(len(metadata)))  # Return all indices
    
    matching_indices = []
    for idx, item in enumerate(metadata):
        # First try to use category_key from metadata (if available from indexer)
        category_key = item.get('category_key')
        
        if category_key:
            # Map category_key to simple category names
            # e.g., mens_watch, womens_watch -> watch
            if category == 'watch' and 'watch' in category_key:
                matching_indices.append(idx)
            elif category == 'sunglasses' and 'sunglasses' in category_key:
                matching_indices.append(idx)
            elif category == 'bag' and ('bag' in category_key or 'handbag' in category_key):
                matching_indices.append(idx)
            elif category == 'shoes' and ('shoe' in category_key or 'loafer' in category_key or 'flipflop' in category_key):
                matching_indices.append(idx)
            elif category == 'wallet' and 'wallet' in category_key:
                matching_indices.append(idx)
            elif category == 'bracelet' and 'bracelet' in category_key:
                matching_indices.append(idx)
        else:
            # Fallback to old method if category_key not in metadata
            product_name = item.get('product_name', '').lower()
            detected_category = detect_category_from_metadata(product_name)
            
            if detected_category == category:
                matching_indices.append(idx)
    
    return matching_indices


def detect_category_from_image(img: Image.Image) -> Optional[str]:
    """
    Simple category detection using CLIP text similarity.
    Compares image with category descriptions to predict category.
    """
    try:
        # Category descriptions for CLIP
        category_texts = [
            "a watch on someone's wrist",
            "a bag or handbag or purse",
            "sunglasses or eyewear",
            "shoes or footwear",
            "a wallet",
            "a bracelet or jewelry"
        ]
        
        # Generate text embeddings for categories
        text_embeddings = model.encode(category_texts, convert_to_numpy=True)
        
        # Generate image embedding
        image_embedding = model.encode(img, convert_to_numpy=True)
        
        # Normalize embeddings
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        
        # Calculate similarity scores
        similarities = np.dot(text_embeddings, image_embedding)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        category_names = ['watch', 'bag', 'sunglasses', 'shoes', 'wallet', 'bracelet']
        detected_category = category_names[best_idx]
        
        # Only return category if confidence is high enough
        # INCREASED threshold to avoid false category detection
        if best_score > 0.35:  # Threshold for category detection (increased from 0.25)
            print(f"  📂 Detected category: {detected_category} (confidence: {best_score:.3f})")
            return detected_category
        else:
            print(f"  ⚠️ Category unclear (best: {detected_category} @ {best_score:.3f}) - searching ALL categories")
            return None  # Return None to search across all categories
            
    except Exception as e:
        print(f"  ⚠️ Category detection failed: {e}")
        return None


def process_uploaded_image(file_content: bytes) -> Image.Image:
    """Process uploaded image file with proper preprocessing."""
    try:
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            raise ValueError(f"Image too large: {size_mb:.2f}MB (max: {MAX_IMAGE_SIZE_MB}MB)")
        
        # Open and convert image
        img = Image.open(BytesIO(file_content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to standard size for consistent embeddings
        # Use LANCZOS for high-quality downsampling
        if img.size != TARGET_IMAGE_SIZE:
            img = img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
            print(f"  🔧 Resized image to {TARGET_IMAGE_SIZE}")
        
        return img
    
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def search_similar_product(query_embedding: np.ndarray, k: int = 50, category_filter: Optional[str] = None) -> Dict:
    """Search for similar products in FAISS index with multi-image voting and category filtering."""
    # Normalize query embedding for cosine similarity
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Get category-filtered indices if category is provided
    if category_filter:
        category_indices = get_category_indices(category_filter)
        print(f"  📂 Category filter: {category_filter} ({len(category_indices)} products)")
        
        # If no products in category, fall back to full search
        if len(category_indices) == 0:
            print(f"  ⚠️ No products found in category {category_filter}, searching all products")
            category_filter = None
    
    # Search in FAISS index (get more results for voting)
    print(f"  🔍 Searching {k} candidates from {index.ntotal} total vectors...")
    scores, indices = index.search(query_embedding, k)
    
    # Check if using IndexFlatIP (Inner Product) or IndexFlatL2
    index_type = type(index).__name__
    print(f"  Index type: {index_type}")
    
    # Collect all results with product grouping
    product_matches = {}  # {product_url: [scores]}
    image_results = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            # Apply category filter if specified
            if category_filter:
                item_category = detect_category_from_metadata(metadata[idx]['product_name'])
                if item_category != category_filter:
                    continue  # Skip products not in the target category
            # Convert score to similarity
            if 'IP' in index_type:
                similarity = float(score)
                l2_distance = np.sqrt(2.0 * (1.0 - similarity))
            else:
                l2_distance = float(score)
                similarity = 1.0 - (l2_distance * l2_distance / 2.0)
            
            product_name = metadata[idx]['product_name']
            product_url = metadata[idx]['product_url']
            
            # Group by product (multiple images per product)
            if product_url not in product_matches:
                product_matches[product_url] = {
                    'product_name': product_name,
                    'product_url': product_url,
                    'price': metadata[idx].get('price', 'N/A'),
                    'scores': [],
                    'best_image': metadata[idx]['image_url'],
                    'best_score': similarity
                }
            
            product_matches[product_url]['scores'].append(similarity)
            
            # Track best image for this product
            if similarity > product_matches[product_url]['best_score']:
                product_matches[product_url]['best_score'] = similarity
                product_matches[product_url]['best_image'] = metadata[idx]['image_url']
            
            # Also keep individual image results for debugging
            image_results.append({
                'product_name': product_name,
                'product_url': product_url,
                'image_url': metadata[idx]['image_url'],
                'price': metadata[idx].get('price', 'N/A'),
                'similarity_score': float(similarity),
                'l2_distance': float(l2_distance),
                'raw_score': float(score)
            })
    
    # Calculate aggregate scores for each product (voting system)
    results = []
    for prod_url, prod_data in product_matches.items():
        scores = prod_data['scores']
        
        # Voting metrics
        avg_score = np.mean(scores)  # Average similarity
        max_score = np.max(scores)   # Best image match
        top3_avg = np.mean(sorted(scores, reverse=True)[:3])  # Top 3 images average
        vote_count = len(scores)     # How many images matched
        
        # Combined score: weighted average
        # Favor products with multiple high-scoring images
        # IMPROVED: Give more weight to best match, less to averages
        combined_score = (max_score * 0.6) + (avg_score * 0.2) + (top3_avg * 0.2)
        
        # Bonus for products with multiple matching images (voting confidence)
        if vote_count >= 3:
            combined_score *= 1.05  # 5% boost for 3+ matching images
        elif vote_count >= 5:
            combined_score *= 1.10  # 10% boost for 5+ matching images
        
        l2_distance = np.sqrt(2.0 * (1.0 - combined_score))
        
        results.append({
            'product_name': prod_data['product_name'],
            'product_url': prod_data['product_url'],
            'image_url': prod_data['best_image'],
            'price': prod_data['price'],
            'similarity_score': float(combined_score),
            'max_score': float(max_score),
            'avg_score': float(avg_score),
            'vote_count': vote_count,
            'l2_distance': float(l2_distance),
            'raw_score': float(combined_score)
        })
    
    # Sort by combined score (highest first)
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return results


@app.post("/search")
async def search_product(file: UploadFile = File(...)):
    """
    Search for similar products using uploaded image.
    
    Returns:
    - Product URL if similarity > threshold
    - "No match found" otherwise
    """
    if model is None or index is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Resources not loaded."
        )
    
    try:
        # Read and process uploaded image
        file_content = await file.read()
        img = process_uploaded_image(file_content)
        
        # Detect category from image
        detected_category = detect_category_from_image(img)
        
        # Generate embedding
        query_embedding = model.encode(img, convert_to_numpy=True)
        
        # Search for similar products with category filter
        # INCREASED k value to get more candidates for voting (especially important with 8000+ products)
        k_value = 100 if detected_category else 150  # More candidates if no category filter
        results = search_similar_product(query_embedding, k=k_value, category_filter=detected_category)
        
        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_match",
                    "message": "No match found",
                    "similarity_score": 0.0
                }
            )
        
        # Get top match
        top_match = results[0]
        similarity_score = top_match['similarity_score']
        l2_distance = top_match['l2_distance']
        
        # Log top 5 results for debugging
        print(f"\n🔍 Search Results (Top 5 Products with Multi-Image Voting):")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r['product_name'][:50]}")
            print(f"      Combined: {r['similarity_score']:.4f} | Max: {r['max_score']:.4f} | Avg: {r['avg_score']:.4f} | Votes: {r['vote_count']}")
        
        # Check if top results are very close in score (ambiguous match)
        if len(results) >= 2:
            score_diff = results[0]['similarity_score'] - results[1]['similarity_score']
            # RELAXED: Only warn if difference is very small (< 3%) AND score is low
            if score_diff < 0.03 and similarity_score < 0.70:
                # Top matches are too close, might be wrong product
                print(f"⚠️ Ambiguous match: Top 2 scores differ by only {score_diff:.4f}")
                print(f"   Best match might not be correct (confidence too low)")
        
        # Apply three-tier threshold system (use similarity score - higher is better)
        if similarity_score >= SIMILARITY_THRESHOLD_HIGH:
            # High confidence (85%+)
            print(f"✅ HIGH confidence match: {similarity_score:.4f} >= {SIMILARITY_THRESHOLD_HIGH}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "match_found",
                    "product_name": top_match['product_name'],
                    "product_url": top_match['product_url'],
                    "price": top_match['price'],
                    "similarity_score": similarity_score,
                    "l2_distance": l2_distance,
                    "matched_image_url": top_match['image_url'],
                    "confidence": "high",
                    "detected_category": detected_category,
                    "top_5_results": results[:5]  # Include top 5 for debugging
                }
            )
        elif similarity_score >= SIMILARITY_THRESHOLD_MEDIUM:
            # Medium confidence (75-85%)
            print(f"✅ MEDIUM confidence match: {similarity_score:.4f} (between {SIMILARITY_THRESHOLD_MEDIUM} and {SIMILARITY_THRESHOLD_HIGH})")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "match_found",
                    "product_name": top_match['product_name'],
                    "product_url": top_match['product_url'],
                    "price": top_match['price'],
                    "similarity_score": similarity_score,
                    "l2_distance": l2_distance,
                    "matched_image_url": top_match['image_url'],
                    "confidence": "medium",
                    "warning": "Good match found. Please verify the product details.",
                    "detected_category": detected_category,
                    "top_5_results": results[:5]  # Include top 5 for debugging
                }
            )
        elif similarity_score >= SIMILARITY_THRESHOLD_LOW:
            # Low confidence (70-75%)
            print(f"⚠️ LOW confidence match: {similarity_score:.4f} (between {SIMILARITY_THRESHOLD_LOW} and {SIMILARITY_THRESHOLD_MEDIUM})")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "match_found",
                    "product_name": top_match['product_name'],
                    "product_url": top_match['product_url'],
                    "price": top_match['price'],
                    "similarity_score": similarity_score,
                    "l2_distance": l2_distance,
                    "matched_image_url": top_match['image_url'],
                    "confidence": "low",
                    "warning": "Possible match found. Please carefully verify before ordering.",
                    "detected_category": detected_category,
                    "top_5_results": results[:5]  # Include top 5 for debugging
                }
            )
        else:
            print(f"❌ Match rejected: {similarity_score:.4f} < {SIMILARITY_THRESHOLD_LOW}")
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_match",
                    "message": f"No confident match found (best similarity: {similarity_score:.3f} < {SIMILARITY_THRESHOLD_LOW})",
                    "similarity_score": similarity_score,
                    "l2_distance": l2_distance,
                    "detected_category": detected_category,
                    "top_5_results": results[:5]  # Include for debugging
                }
            )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/similar")
async def find_similar_products(file: UploadFile = File(...)):
    """
    Find similar products by style (not exact match).
    Returns top 10 visually similar products regardless of exact match.
    
    This is useful for:
    - "Show me similar bags"
    - Product recommendations
    - Upselling alternatives
    """
    if model is None or index is None or metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Resources not loaded."
        )
    
    try:
        # Read and process uploaded image
        file_content = await file.read()
        img = process_uploaded_image(file_content)
        
        # Detect category from image
        detected_category = detect_category_from_image(img)
        
        # Generate embedding
        query_embedding = model.encode(img, convert_to_numpy=True)
        
        # Search for similar products with category filter (get more results)
        results = search_similar_product(query_embedding, k=100, category_filter=detected_category)
        
        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_results",
                    "message": "No similar products found",
                    "detected_category": detected_category
                }
            )
        
        # Return top 10 similar products (lower threshold than exact match)
        similar_products = []
        for r in results[:10]:
            # Include products with similarity > 0.70 (lower than exact match threshold)
            if r['similarity_score'] >= 0.70:
                similar_products.append({
                    'product_name': r['product_name'],
                    'product_url': r['product_url'],
                    'price': r['price'],
                    'similarity_score': r['similarity_score'],
                    'image_url': r['image_url']
                })
        
        if not similar_products:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_results",
                    "message": "No similar products found with sufficient similarity",
                    "detected_category": detected_category
                }
            )
        
        print(f"\n🎨 Found {len(similar_products)} similar products")
        for i, p in enumerate(similar_products[:5], 1):
            print(f"  {i}. {p['product_name'][:50]} - {p['similarity_score']:.3f}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "count": len(similar_products),
                "detected_category": detected_category,
                "similar_products": similar_products,
                "message": f"Found {len(similar_products)} similar products"
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "image_identifier_api",
        "port": 8002,
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "indexed_products": len(metadata) if metadata else 0
    }

if __name__ == "__main__":
    print("=" * 70)
    print("  VISUAL SEARCH ENGINE - API SERVER")
    print("  Starting FastAPI server on port 8002...")
    print("=" * 70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
