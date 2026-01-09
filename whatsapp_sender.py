"""
WhatsApp Product Sender - Single File Solution
Send product images directly to WhatsApp based on user queries
"""

import requests
from typing import Dict, List
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Load environment variables
load_dotenv()

# =====================================================
# CONFIGURATION - Update these values
# =====================================================
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://72.62.76.36:8080")
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "whatsapp bot")
API_KEY = os.getenv("EVOLUTION_API_KEY", "ayushpatel143597")

# Local Product Search API
SEARCH_API_URL = os.getenv("TEXT_SEARCH_API_URL", "http://text_search_api:8001")

# =====================================================
# CORE FUNCTIONS
# =====================================================

def search_products(query: str, max_results: int = 5) -> Dict:
    """Search products in database and get images."""
    try:
        response = requests.post(
            f"{SEARCH_API_URL}/search/images-list",
            json={"query": query, "max_results": max_results},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "products": []}
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        return {"status": "error", "products": []}


def send_whatsapp_image(phone_number: str, image_base64: str, caption: str = "") -> Dict:
    """Send image to WhatsApp via Evolution API."""
    try:
        url = f"{EVOLUTION_API_URL}/message/sendMedia/{INSTANCE_NAME}"
        
        headers = {
            "Content-Type": "application/json",
            "apikey": API_KEY
        }
        
        payload = {
            "number": phone_number,
            "mediatype": "image",
            "media": image_base64,
            "caption": caption
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        return {
            "success": response.status_code in [200, 201],
            "caption": caption
        }
    
    except Exception as e:
        print(f"❌ Send error: {e}")
        return {
            "success": False,
            "caption": caption
        }


def send_single_product_images(phone_number: str, product: Dict, product_idx: int, total_products: int) -> Dict:
    """Send all images for a single product in parallel."""
    product_name = product['product_name']
    price = product['price']
    images_base64 = product.get('images_base64', [])
    
    if not images_base64:
        return {
            "product_name": product_name,
            "success": False,
            "images_sent": 0,
            "total_images": 0
        }
    
    print(f"[{product_idx}/{total_products}] 📦 {product_name}")
    print(f"              💰 ₹{price}")
    print(f"              📸 Sending {len(images_base64)} images in parallel...")
    
    # Send all images for this product in parallel
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for img_idx, img_base64 in enumerate(images_base64, 1):
            # Only add caption to first image
            caption = f"📦 {product_name}\n💰 ₹{price}\n📸 Image {img_idx}/{len(images_base64)}" if img_idx == 1 else f"📸 {img_idx}/{len(images_base64)}"
            future = executor.submit(send_whatsapp_image, phone_number, img_base64, caption)
            futures.append(future)
        
        # Wait for all images to be sent
        for future in as_completed(futures):
            try:
                result = future.result()
                if result['success']:
                    success_count += 1
            except Exception as e:
                print(f"              ❌ Error: {e}")
    
    print(f"              ✅ Sent {success_count}/{len(images_base64)} images\n")
    
    return {
        "product_name": product_name,
        "success": success_count > 0,
        "images_sent": success_count,
        "total_images": len(images_base64)
    }


def send_whatsapp_text(phone_number: str, message: str) -> bool:
    """Send text message to WhatsApp."""
    try:
        url = f"{EVOLUTION_API_URL}/message/sendText/{INSTANCE_NAME}"
        
        headers = {
            "Content-Type": "application/json",
            "apikey": API_KEY
        }
        
        payload = {
            "number": phone_number,
            "text": message
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        return response.status_code in [200, 201]
    
    except Exception as e:
        print(f"❌ Text send error: {e}")
        return False


def send_products_to_whatsapp(phone_number: str, user_query: str, max_products: int = 5) -> bool:
    """
    Main function: Search products and send ALL images to WhatsApp in parallel.
    
    Args:
        phone_number: WhatsApp number with country code (e.g., "919876543210")
        user_query: Search query (e.g., "rolex watch", "michael bag")
        max_products: Maximum number of products to send (default: 5)
    
    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print(f"📱 Sending to: {phone_number}")
    print(f"🔍 Query: {user_query}")
    print("="*70 + "\n")
    
    # Clean query (remove common words)
    query = user_query.lower()
    remove_words = ["chahiye", "dikhao", "show", "me", "muje", "mujhe", "ki", "ka", "ke", "de", "do"]
    for word in remove_words:
        query = query.replace(word, "")
    query = query.strip()
    
    print(f"🔍 Searching for: '{query}'")
    
    # Search products
    result = search_products(query, max_products)
    
    if result['status'] != 'success' or len(result['products']) == 0:
        print(f"❌ No products found for '{query}'\n")
        send_whatsapp_text(
            phone_number,
            f"😕 Sorry, no products found matching '{query}'.\n\nTry:\n• Brand names (Rolex, Louis Vuitton)\n• Product types (watch, bag)"
        )
        return False
    
    total_products = len(result['products'])
    total_images = result.get('total_images', 0)
    
    print(f"✅ Found {total_products} products with {total_images} total images\n")
    
    # Send initial message
    send_whatsapp_text(
        phone_number,
        f"✅ Found {total_products} products with {total_images} images matching '{query}'!\n\n⚡ Sending all images in parallel..."
    )
    
    # Send all products in parallel (each product sends its images in parallel too)
    total_images_sent = 0
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for idx, product in enumerate(result['products'], 1):
            future = executor.submit(send_single_product_images, phone_number, product, idx, total_products)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                total_images_sent += result['images_sent']
            except Exception as e:
                print(f"❌ Error processing product: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Send completion message
    send_whatsapp_text(
        phone_number,
        f"✅ Sent {total_images_sent} images from {len(results)} products in {elapsed_time:.1f} seconds!\n\nNeed more options? Just ask!"
    )
    
    print("="*70)
    print(f"✅ Complete! Sent {total_images_sent} images in {elapsed_time:.1f} seconds")
    print("="*70 + "\n")
    
    return total_images_sent > 0


# =====================================================
# USAGE EXAMPLES
# =====================================================

if __name__ == "__main__":
    import sys
    
    # Example 1: Command line usage
    if len(sys.argv) >= 3:
        phone = sys.argv[1]
        query = " ".join(sys.argv[2:])
        send_products_to_whatsapp(phone, query)
    
    # Example 2: Direct call
    else:
        print("\n" + "="*70)
        print("🤖 WhatsApp Product Sender")
        print("="*70)
        print("\nUsage:")
        print("  python whatsapp_sender.py <phone_number> <query>")
        print("\nExamples:")
        print("  python whatsapp_sender.py 919016220667 michael bag")
        print("  python whatsapp_sender.py 919016220667 rolex watch")
        print("  python whatsapp_sender.py 919016220667 louis vuitton")
        print("\nOr import and use:")
        print("  from whatsapp_sender import send_products_to_whatsapp")
        print("  send_products_to_whatsapp('919016220667', 'michael bag')")
        print("="*70 + "\n")
        
        # Test with default values
        print("Running test with default values...\n")
        send_products_to_whatsapp("919016220667", "michael")
