"""
Visual Search Engine - Indexer
Downloads images from MongoDB, generates CLIP embeddings, and creates FAISS index.
Optimized for CPU-only VPS with 2GB RAM.
"""

import os
import time
import pickle
import requests
from io import BytesIO
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import faiss
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "watchvine_refined")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")
MAX_WORKERS = 5  # Limited for 2GB RAM VPS
MODEL_NAME = "clip-ViT-B-32"
INDEX_FILE = "vector_index.bin"
METADATA_FILE = "metadata.pkl"
HASH_INDEX_FILE = "hash_index.pkl"  # New: perceptual hash index
RETRY_DELAY = 2
TARGET_IMAGE_SIZE = (224, 224)  # Standard CLIP input size for consistency



class ImageDownloader:
    """Handle image downloads with retries and timeout."""
    
    @staticmethod
    def download_image(url: str, retries: int = 3) -> Image.Image:
        """Download and return PIL Image with retry logic and preprocessing."""
        for attempt in range(retries):
            try:
                response = requests.get(
                    url, 
                    timeout=30,  # 30 seconds timeout
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard size for consistent embeddings
                # This is CRITICAL - must match the size used during search
                if img.size != TARGET_IMAGE_SIZE:
                    img = img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
                
                return img
            
            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1}/{retries} failed for {url}: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
        
        return None


class VectorIndexer:
    """Main indexer class for creating FAISS index from MongoDB images."""
    
    def __init__(self):
        """Initialize MongoDB connection and CLIP model."""
        print("🚀 Initializing Visual Search Indexer...")
        
        # Connect to MongoDB
        print(f"📦 Connecting to MongoDB: {MONGO_URI}")
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Load CLIP model
        print(f"🤖 Loading CLIP model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        
        # Get embedding dimension - handle None case
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim is None:
            # For CLIP models, dimension is typically 512
            # We'll get it from a test encoding
            print(f"⚠️  Embedding dimension is None, detecting from model...")
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.embedding_dim = len(test_embedding)
        
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        
        self.downloader = ImageDownloader()
        self.embeddings = []
        self.metadata = []
        self.hash_index = {}  # New: perceptual hash index {index: hash_string}
    
    def fetch_products(self) -> List[Dict]:
        """Fetch products from MongoDB."""
        print(f"\n📊 Fetching all products from MongoDB...")
        products = list(self.collection.find())
        print(f"✅ Found {len(products)} products")
        return products
    
    def download_and_encode_image(self, url: str, product_name: str, 
                                   product_url: str, price: str = "N/A", 
                                   category: str = None, category_key: str = None) -> Tuple[np.ndarray, Dict, str]:
        """Download single image, generate embedding and perceptual hash."""
        try:
            img = self.downloader.download_image(url)
            
            # Generate CLIP embedding
            embedding = self.model.encode(img, convert_to_numpy=True)
            
            # Generate perceptual hash (pHash) for exact match detection
            perceptual_hash = str(imagehash.phash(img, hash_size=16))
            
            metadata = {
                'product_name': product_name,
                'product_url': product_url,
                'image_url': url,
                'price': price,
                'category': category,
                'category_key': category_key
            }
            
            return embedding, metadata, perceptual_hash
        
        except Exception as e:
            print(f"  ⚠️  Skipping image {url}: {str(e)}")
            return None, None, None
    
    def process_products(self, products: List[Dict]):
        """Process all products and their images."""
        print(f"\n🔄 Processing products and generating embeddings...")
        
        total_images = 0
        successful_embeddings = 0
        
        for idx, product in enumerate(products, 1):
            # Support both field name formats
            product_name = product.get('product_name') or product.get('name', 'Unknown')
            product_url = product.get('product_url') or product.get('url', '')
            image_urls = product.get('image_urls', [])
            price = product.get('price', 'N/A')
            category = product.get('category')
            category_key = product.get('category_key')
            
            print(f"\n[{idx}/{len(products)}] 📦 Product: {product_name}")
            if category:
                print(f"  📂 Category: {category} ({category_key})")
            print(f"  🖼️  Images to process: {len(image_urls)}")
            print(f"  💰 Price: ₹{price}")
            
            if not image_urls:
                print("  ⚠️  No images found for this product")
                continue
            
            total_images += len(image_urls)
            
            # Process images with threading (limited workers for low RAM)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        self.download_and_encode_image,
                        url, product_name, product_url, price, category, category_key
                    ): url for url in image_urls
                }
                
                for future in as_completed(futures):
                    embedding, metadata, perceptual_hash = future.result()
                    
                    if embedding is not None and metadata is not None and perceptual_hash is not None:
                        # Store embedding and metadata
                        current_index = len(self.embeddings)
                        self.embeddings.append(embedding)
                        self.metadata.append(metadata)
                        
                        # Store perceptual hash with index
                        self.hash_index[current_index] = perceptual_hash
                        
                        successful_embeddings += 1
                        print(f"  ✅ Processed: {metadata['image_url'][:60]}... (hash: {perceptual_hash[:16]}...)")
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.1)
        
        print(f"\n📈 Summary:")
        print(f"  Total images attempted: {total_images}")
        print(f"  Successful embeddings: {successful_embeddings}")
        print(f"  Failed: {total_images - successful_embeddings}")
    
    def create_faiss_index(self):
        """Create FAISS index from embeddings."""
        print(f"\n🔨 Creating FAISS index...")
        
        if not self.embeddings:
            raise ValueError("No embeddings to index!")
        
        # Convert to numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Get actual dimension from embeddings if self.embedding_dim is None
        actual_dim = embeddings_array.shape[1]
        print(f"  Embedding dimension: {actual_dim}")
        
        # Normalize vectors for cosine similarity (IndexFlatIP with normalized vectors)
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index (Inner Product) - use actual dimension
        index = faiss.IndexFlatIP(int(actual_dim))
        index.add(embeddings_array)
        
        print(f"✅ FAISS index created with {index.ntotal} vectors")
        return index
    
    def save_index_and_metadata(self, index):
        """Save FAISS index, metadata, and perceptual hash index to disk."""
        print(f"\n💾 Saving index, metadata, and hash index...")
        
        # Save FAISS index
        faiss.write_index(index, INDEX_FILE)
        print(f"✅ FAISS index saved to: {INDEX_FILE}")
        
        # Save metadata
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"✅ Metadata saved to: {METADATA_FILE}")
        
        # Save perceptual hash index
        with open(HASH_INDEX_FILE, 'wb') as f:
            pickle.dump(self.hash_index, f)
        print(f"✅ Hash index saved to: {HASH_INDEX_FILE} ({len(self.hash_index)} hashes)")
        
        # Print file sizes
        index_size = os.path.getsize(INDEX_FILE) / 1024
        metadata_size = os.path.getsize(METADATA_FILE) / 1024
        hash_index_size = os.path.getsize(HASH_INDEX_FILE) / 1024
        print(f"\n📊 File sizes:")
        print(f"  {INDEX_FILE}: {index_size:.2f} KB")
        print(f"  {METADATA_FILE}: {metadata_size:.2f} KB")
        print(f"  {HASH_INDEX_FILE}: {hash_index_size:.2f} KB")
    
    def run(self):
        """Main execution flow."""
        start_time = time.time()
        
        try:
            # Fetch products
            products = self.fetch_products()
            
            if not products:
                print("❌ No products found in MongoDB!")
                return
            
            # Process products and generate embeddings
            self.process_products(products)
            
            if not self.embeddings:
                print("❌ No embeddings generated. Check image URLs and connectivity.")
                return
            
            # Create FAISS index
            index = self.create_faiss_index()
            
            # Save to disk
            self.save_index_and_metadata(index)
            
            elapsed_time = time.time() - start_time
            print(f"\n🎉 Indexing completed successfully in {elapsed_time:.2f} seconds!")
            print(f"✅ Ready for search API!")
            
        except Exception as e:
            print(f"\n❌ Error during indexing: {str(e)}")
            raise
        
        finally:
            self.client.close()
            print("\n🔒 MongoDB connection closed")


if __name__ == "__main__":
    print("=" * 70)
    print("  VISUAL SEARCH ENGINE - INDEXER")
    print("  Optimized for CPU-only VPS (2GB RAM)")
    print("=" * 70)
    
    indexer = VectorIndexer()
    indexer.run()
