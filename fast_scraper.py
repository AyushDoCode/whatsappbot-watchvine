"""
Fast Multi-threaded Scraper for Large Product Catalogs
Optimized for scraping thousands of products quickly
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import os
import random
import hashlib
import secrets
from pymongo import MongoClient
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import signal
import sys

# --- CONFIGURATION ---
DB_NAME = "watchvine_refined"
COLLECTION_NAME = "products"
BASE_URL = "https://watchvine01.cartpe.in/"
LOAD_MORE_URL = "https://watchvine01.cartpe.in/store_product_loadmore"
WEB_TOKEN = None  # Will be extracted dynamically from the page

# Extended User Agents with more diversity to avoid detection
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    
    # Mobile browsers (sometimes helps)
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
]

def get_random_headers(user_agent=None):
    """Generate random headers to avoid detection with more realistic variations"""
    if user_agent is None:
        user_agent = random.choice(USER_AGENTS)
    
    # Vary accept headers slightly for more realism
    accept_formats = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    ]
    
    accept_languages = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9,en-US;q=0.8",
        "en-US,en;q=0.9,hi;q=0.8",
        "en-IN,en;q=0.9,hi;q=0.8",
    ]
    
    return {
        "User-Agent": user_agent,
        "Accept": random.choice(accept_formats),
        "Accept-Language": random.choice(accept_languages),
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

HEADERS = get_random_headers()


def generate_session_cookies():
    """Generate realistic session cookies that don't expire"""
    current_timestamp = int(time.time())
    
    # Generate random cookie_set_user (like bs_694d1152af500)
    random_hash = secrets.token_hex(6)  # 12 character hex
    cookie_set_user = f"bs_{random_hash}"
    
    # Generate ci_session (26 character alphanumeric)
    ci_session = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=26))
    
    # AWS load balancer cookies - generate realistic looking values
    aws_token = secrets.token_urlsafe(64)[:88]  # Similar length to real AWS cookies
    
    cookies = {
        'store_session': str(current_timestamp),
        'cookie_set_user': cookie_set_user,
        'AWSALB': aws_token,
        'AWSALBCORS': aws_token,
        'ci_session': ci_session
    }
    
    return cookies


def get_cookie_string(cookies_dict):
    """Convert cookie dict to cookie header string"""
    return '; '.join([f"{k}={v}" for k, v in cookies_dict.items()])

# Category-wise URLs for accurate product categorization
CATEGORIES = {
    "mens_sunglasses": {
        "name": "Men's Sunglasses & Eyewear",
        "url": "sunglasses-eye-wear-men.html",
        "slug": "sunglasses-eye-wear-men"
    },
    "womens_sunglasses": {
        "name": "Women's Sunglasses & Eyewear",
        "url": "sunglasses-eye-wear-women.html",
        "slug": "sunglasses-eye-wear-women"
    },
    "mens_watch": {
        "name": "Men's Watch",
        "url": "mens-watch.html",
        "slug": "mens-watch"
    },
    "womens_watch": {
        "name": "Women's Watch",
        "url": "ladies-watch-watches.html",
        "slug": "ladies-watch-watches"
    },
    "wallet": {
        "name": "Wallet",
        "url": "wallet.html",
        "slug": "wallet"
    },
    "handbag": {
        "name": "Hand Bag",
        "url": "hand-bags.html",
        "slug": "hand-bags"
    },
    "premium_sunglasses": {
        "name": "Premium Sunglasses",
        "url": "premium-sunglass-.html",
        "slug": "premium-sunglass-"
    },
    "flipflops": {
        "name": "Flip Flops",
        "url": "flipflops-footwear.html",
        "slug": "flipflops-footwear"
    },
    "loafers": {
        "name": "Formals & Loafers",
        "url": "loafers.html",
        "slug": "loafers"
    },
    "mens_shoes": {
        "name": "Men's Shoes",
        "url": "men-rsquo-s-shoe-footwear.html",
        "slug": "men-rsquo-s-shoe-footwear"
    },
    "womens_shoes": {
        "name": "Women's Shoes",
        "url": "ladies-shoes-footwear-women.html",
        "slug": "ladies-shoes-footwear-women"
    },
    "premium_shoes": {
        "name": "Premium Shoes",
        "url": "premium-shoes-footwear.html",
        "slug": "premium-shoes-footwear"
    },
    "bracelet": {
        "name": "Bracelet & Jewellery",
        "url": "bracellet-jewellery.html",
        "slug": "bracellet-jewellery"
    },
    "loafers_men": {
        "name": "Men's Loafers",
        "url": "loafers-footwear-men-footwear.html",
        "slug": "loafers-footwear-men-footwear"
    }
}

# Performance settings - Optimized for SPEED with stability
MAX_WORKERS = 5  # Increased threads for faster scraping
BATCH_SIZE = 100  # Save to DB every X products
REQUEST_TIMEOUT = 30  # Reduced timeout for faster failures
RETRY_ATTEMPTS = 3  # Reduced retries for faster recovery
RATE_LIMIT_DELAY = 0.5  # Minimal delay between requests (0.5-1 seconds)
RETRY_DELAY_MIN = 3  # Reduced retry delay (seconds)
RETRY_DELAY_MAX = 8  # Reduced maximum retry delay (seconds)
EXPONENTIAL_BACKOFF = True  # Use exponential backoff on retries
COOKIE_REFRESH_INTERVAL = 600  # Refresh cookies every 10 minutes (cookies expire in 3 days)
REQUESTS_PER_COOKIE_CHECK = 50  # Check every 50 requests (less frequent)

# Global counters with thread-safe lock
stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'start_time': None
}
stats_lock = Lock()
products_buffer = []
buffer_lock = Lock()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Interrupt received! Saving progress...")
    if products_buffer:
        save_batch_to_db(products_buffer)
    print(f"✅ Saved {stats['success']} products before exit")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def extract_price(soup):
    """Enhanced price extraction."""
    price = "Price Not Found"
    
    price_tag = soup.find(lambda tag: tag.name in ['h4', 'span', 'div', 'p'] and '₹' in tag.get_text())
    
    if not price_tag:
        price_tag = soup.find(class_=re.compile(r'price|current-price|product-price', re.I))
    
    if price_tag:
        raw_price = price_tag.get_text(separator=" ", strip=True)
        strike = price_tag.find(['strike', 's', 'del']) or price_tag.find('span', style=re.compile(r'line-through'))
        if strike:
            strike_text = strike.get_text(strip=True)
            raw_price = raw_price.replace(strike_text, "").strip()
        
        price_match = re.search(r'₹\s?[\d,.]+', raw_price)
        if price_match:
            price = price_match.group(0).replace(" ", "").replace("₹", "").strip()
        else:
            price = raw_price
    
    return price


def extract_images(soup, product_url):
    """Enhanced image extraction with universal parser approach."""
    image_urls = []
    
    # Remove unwanted sections
    for unwanted in soup.find_all(['div', 'section']):
        if any(term in unwanted.get_text() for term in ["Related Products", "Recently Viewed", "You may also like"]):
            unwanted.decompose()
    
    # Look for product-specific containers first
    containers = soup.find_all('div', class_=['product-slider', 'owl-item', 'item', 'product-details-img', 
                                               'product-gallery', 'image-gallery', 'product-images'])
    target_areas = containers if containers else [soup]
    
    for area in target_areas:
        for img in area.find_all('img'):
            # Try multiple image source attributes
            src = img.get('data-src') or img.get('src') or img.get('data-lazy') or img.get('data-original')
            if src:
                full_url = urljoin(product_url, src)
                # Filter out logos, icons, and duplicates
                if ("logo" not in full_url.lower() and 
                    "icon" not in full_url.lower() and
                    full_url not in image_urls and
                    any(x in full_url for x in ["/product/", "/images/", "uploads", "media", "gallery"])):
                    image_urls.append(full_url)
    
    return image_urls


def universal_product_parser(html_content):
    """Universal parser that finds products even if class names change.
    
    This is an alternative parsing method that's more resilient to HTML structure changes.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    products = []
    
    # Find all potential product containers
    # Cartpe usually uses 'col-' based layout or 'item' classes
    containers = soup.find_all(['div', 'li', 'a'], class_=True)
    
    for box in containers:
        # Skip parent containers - we only want the deepest product containers
        if box.find(['div', 'li'], class_=True):
            continue
        
        text = box.get_text(strip=True)
        
        # Check if this element contains a price (indicates it's a product)
        if re.search(r'[₹|Rs|RS]', text):
            try:
                # Find product name: Usually in h3, h4, h5, p, or a tag
                name = "Product"
                name_elem = box.find(['h5', 'h4', 'h3', 'p', 'a'])
                if name_elem:
                    name = name_elem.get_text(strip=True)
                
                # Find price
                price_match = re.search(r'([₹|Rs|RS]\s?[\d,.]+)', text)
                price = price_match.group(1) if price_match else "N/A"
                
                # Find product URL
                link = box.find('a', href=True)
                url = link['href'] if link else ""
                
                # Find image
                img = box.find('img')
                img_url = img.get('src') or img.get('data-src') if img else ""
                
                if name and len(name) > 3 and url:
                    products.append({
                        "name": name,
                        "price": price,
                        "url": url,
                        "image": img_url
                    })
            except:
                continue
    
    # Remove duplicates based on name
    unique_products = list({p['name']: p for p in products}.values())
    return unique_products


def scrape_single_product(product_data):
    """Scrape a single product (thread-safe) with intelligent retry and session management."""
    p_url, p_name, category_key, category_name, index, total = product_data
    
    session = None
    try:
        # Create a fresh session for this product
        session = requests.Session()
        user_agent = random.choice(USER_AGENTS)
        
        # Fetch product page with retry and smart backoff
        response = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # Use consistent user agent for the session
                headers = get_random_headers(user_agent=user_agent)
                
                response = session.get(p_url, headers=headers, timeout=REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    break
                elif response.status_code == 403:
                    # 403 Forbidden - use exponential backoff
                    if EXPONENTIAL_BACKOFF:
                        retry_delay = min(RETRY_DELAY_MAX, RETRY_DELAY_MIN * (2 ** attempt)) + random.uniform(0, 4)
                    else:
                        retry_delay = random.uniform(RETRY_DELAY_MIN, RETRY_DELAY_MAX)
                    
                    print(f"   ⚠️  Product [{p_name[:30]}...] - 403 on attempt {attempt+1}/{RETRY_ATTEMPTS}, waiting {retry_delay:.1f}s")
                    time.sleep(retry_delay)
                    
                    # Create new session with different user agent
                    session.close()
                    session = requests.Session()
                    user_agent = random.choice(USER_AGENTS)
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    retry_delay = random.uniform(20, 30) * (attempt + 1)
                    print(f"   ⚠️  Product [{p_name[:30]}...] - 429 Rate Limited, waiting {retry_delay:.1f}s")
                    time.sleep(retry_delay)
                else:
                    # Other error - short wait
                    time.sleep(random.uniform(2, 4))
                    
            except requests.exceptions.Timeout:
                print(f"   ⚠️  Product [{p_name[:30]}...] - Timeout on attempt {attempt+1}")
                if attempt == RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(random.uniform(5, 10))
            except requests.exceptions.RequestException as e:
                print(f"   ⚠️  Product [{p_name[:30]}...] - Request error: {str(e)[:40]}")
                if attempt == RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(random.uniform(3, 7))
        
        # Check if we got a valid response
        if response is None or response.status_code != 200:
            raise Exception(f"Failed after {RETRY_ATTEMPTS} attempts - status: {response.status_code if response else 'None'}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data
        price = extract_price(soup)
        image_urls = extract_images(soup, p_url)
        
        result = {
            "name": p_name,
            "url": p_url,
            "price": price,
            "image_urls": image_urls,
            "category": category_name,
            "category_key": category_key,
            "scraped_at": time.time()
        }
        
        # Update stats
        with stats_lock:
            stats['success'] += 1
            elapsed = time.time() - stats['start_time']
            rate = stats['success'] / elapsed if elapsed > 0 else 0
            remaining = total - (stats['success'] + stats['failed'])
            eta = remaining / rate if rate > 0 else 0
            
            print(f"[{stats['success'] + stats['failed']}/{total}] ✓ [{category_name}] {p_name[:40]}... | Price: {price} | Images: {len(image_urls)} | Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m")
        
        # Add to buffer
        with buffer_lock:
            products_buffer.append(result)
            
            # Save batch if buffer is full
            if len(products_buffer) >= BATCH_SIZE:
                save_batch_to_db(products_buffer.copy())
                products_buffer.clear()
        
        # Minimal delay between product scrapes
        time.sleep(random.uniform(RATE_LIMIT_DELAY, RATE_LIMIT_DELAY + 0.5))
        return result
        
    except Exception as e:
        with stats_lock:
            stats['failed'] += 1
            print(f"[{stats['success'] + stats['failed']}/{total}] ✗ [{category_name}] {p_name[:40]}... | Error: {str(e)[:50]}")
        return None
    finally:
        # Always close session
        if session:
            try:
                session.close()
            except:
                pass


def save_batch_to_db(batch):
    """Save a batch of products to MongoDB."""
    if not batch:
        return
    
    try:
        MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        products_col = db[COLLECTION_NAME]
        
        products_col.insert_many(batch, ordered=False)
        print(f"💾 Saved batch of {len(batch)} products to MongoDB")
        
        client.close()
    except Exception as e:
        print(f"❌ Error saving batch to DB: {e}")


def get_category_id(category_url):
    """Extract category ID from the category page for load more functionality."""
    try:
        headers = get_random_headers()
        response = requests.get(category_url, headers=headers, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the category ID in scripts or hidden inputs
        # Method 1: Check for hidden input with cat_id
        cat_input = soup.find('input', {'name': 'cat_ids'})
        if cat_input and cat_input.get('value'):
            return cat_input.get('value')
        
        # Method 2: Check in the URL or page source for category ID patterns
        page_source = str(soup)
        cat_id_match = re.search(r'cat_ids["\']?\s*[:=]\s*["\']?(\d+)', page_source)
        if cat_id_match:
            return cat_id_match.group(1)
        
        # Method 3: Extract from the load more button/script
        load_more_script = soup.find_all('script', string=re.compile(r'cat_ids'))
        for script in load_more_script:
            cat_id_match = re.search(r'cat_ids["\']?\s*[:=]\s*["\']?(\d+)', script.string)
            if cat_id_match:
                return cat_id_match.group(1)
        
        return ""  # Return empty string if not found (means all categories)
        
    except Exception as e:
        print(f"⚠️  Could not extract category ID: {e}")
        return ""


def refresh_session_cookies(session, category_url, user_agent):
    """Refresh session by visiting category page to get fresh cookies from server.
    
    Returns:
        bool: True if refresh successful, False otherwise
    """
    try:
        print(f"   🔄 Refreshing session cookies from server...")
        
        # Create new headers for this request
        headers = get_random_headers(user_agent=user_agent)
        
        # Visit category page to get fresh cookies
        response = session.get(category_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print(f"   ✅ Cookies refreshed successfully: {len(session.cookies)} cookies")
            for cookie in session.cookies:
                print(f"      - {cookie.name}: {cookie.value[:20]}...")
            return True
        elif response.status_code == 403:
            print(f"   ⚠️  Got 403 while refreshing cookies")
            return False
        else:
            print(f"   ⚠️  Unexpected status {response.status_code} while refreshing")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Error refreshing cookies: {e}")
        return False


def extract_web_token(html_content):
    """Extract web_token dynamically from the page source.
    
    Args:
        html_content: HTML content from the page
        
    Returns:
        str: Extracted web_token or None if not found
    """
    try:
        # Regex for common web_token patterns
        # Pattern 1: web_token: "..." or web_token = "..."
        match = re.search(r'web_token["\']?\s*[:=]\s*["\']([a-f0-9]{32,})["\']', html_content)
        if match:
            return match.group(1)
        
        # Pattern 2: name="web_token" value="..."
        match = re.search(r'name="web_token"\s+value="([a-f0-9]{32,})"', html_content)
        if match:
            return match.group(1)
        
        # Pattern 3: data-token or similar attributes
        match = re.search(r'data-token["\']?\s*[:=]\s*["\']([a-f0-9]{32,})["\']', html_content)
        if match:
            return match.group(1)
            
        return None
    except Exception as e:
        print(f"   ⚠️  Error extracting web_token: {e}")
        return None


def create_fresh_session(category_url):
    """Create a completely fresh session with cookies and web_token from server.
    
    Returns:
        tuple: (session, user_agent, web_token) or (None, None, None) if failed
    """
    try:
        session = requests.Session()
        user_agent = random.choice(USER_AGENTS)
        
        # Initial page visit to get cookies AND web_token
        headers = get_random_headers(user_agent=user_agent)
        session.headers.update(headers)
        
        print(f"   🌐 Visiting {category_url[:50]}... to get fresh session...")
        response = session.get(category_url, timeout=30)
        
        if response.status_code == 200:
            # Extract web_token from page source
            web_token = extract_web_token(response.text)
            
            if web_token:
                print(f"   🔑 Fresh web_token extracted: {web_token[:20]}...")
            else:
                print(f"   ⚠️  Could not extract web_token from page")
            
            return session, user_agent, web_token
        else:
            print(f"   ⚠️  Got status {response.status_code}")
            session.close()
            return None, None, None
            
    except Exception as e:
        print(f"   ⚠️  Error creating fresh session: {e}")
        return None, None, None


def scrape_category(category_key, category_info, limit_per_category=None):
    """Scrape ALL products from a specific category using load more API with session.
    
    Enhanced with intelligent cookie refresh mechanism and dynamic web_token extraction.
    """
    category_name = category_info['name']
    category_url = urljoin(BASE_URL, category_info['url'])
    category_slug = category_info['slug']
    
    print(f"\n{'='*80}")
    print(f"📂 Scraping Category: {category_name}")
    print(f"🔗 URL: {category_url}")
    print(f"📋 Slug: {category_slug}")
    print(f"{'='*80}\n")
    
    # Track cookie age to proactively refresh
    session_start_time = time.time()
    last_cookie_refresh = session_start_time
    # Note: ci_session cookie expires after 3 DAYS (259200 seconds)
    # We refresh every 10 minutes to be safe, but cookies won't expire during scraping
    
    # Create initial session and get cookies + web_token from server
    print(f"🔑 Creating fresh session and extracting web_token...")
    time.sleep(random.uniform(0.5, 1))  # Minimal delay
    
    session, user_agent, web_token = create_fresh_session(category_url)
    if not session:
        print(f"❌ Failed to create initial session for {category_name}")
        return []
    
    if not web_token:
        print(f"⚠️  Warning: Could not extract web_token, using fallback")
        # Fallback to hardcoded token if extraction fails
        web_token = "5437338f07f857d9b31727cb3baae61d22f1b08c78d05ed8285b1b5cc7b367083b3e62459f428a9586025b55ecdfd5117a3b38886bbc5858b28ff3eecab6ce93"
    
    try:
        # Server has now SET all required cookies automatically!
        print(f"✅ Got cookies from server: {len(session.cookies)} total")
        for cookie in session.cookies:
            print(f"   - {cookie.name}: {cookie.value[:20]}...")
        
        # Parse initial page
        response = session.get(category_url, timeout=60)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all product divs from initial page
        all_divs = soup.find_all('div', class_=re.compile(r'col-xs-6'))
        initial_count = len(all_divs)
        print(f"✅ Initial page: {initial_count} products loaded")
        
        # Now use load more API to fetch remaining products
        offset = initial_count
        max_attempts = 3
        consecutive_failures = 0
        max_consecutive_failures = 2  # Allow 2 consecutive failures before giving up
        request_count = 0  # Track number of requests made
        
        while True:
            # PROACTIVE COOKIE REFRESH - refresh before they expire!
            current_time = time.time()
            time_since_last_refresh = current_time - last_cookie_refresh
            
            # Refresh cookies every 10 minutes OR every 50 requests
            if time_since_last_refresh >= COOKIE_REFRESH_INTERVAL or (request_count > 0 and request_count % REQUESTS_PER_COOKIE_CHECK == 0):
                print(f"🔄 Proactive cookie refresh (age: {time_since_last_refresh:.0f}s, requests: {request_count})...")
                if refresh_session_cookies(session, category_url, user_agent):
                    last_cookie_refresh = current_time
                    time.sleep(random.uniform(0.3, 0.7))  # Minimal wait after refresh
            
            print(f"📥 Loading more products from offset {offset}...")
            
            # Minimal delay between load more requests
            time.sleep(random.uniform(0.8, 1.5))
            
            # Prepare POST data for load more - exactly as browser sends it
            post_data = {
                'getresult': str(offset),
                'category_slug': category_slug,
                'orderby': 'featured',
                'min_price': '0',
                'max_price': '20000',
                'size_ids': '',
                'variant_status': '0',
                'web_token': web_token,  # Use dynamically extracted token
            }
            
            loaded_in_this_iteration = False
            
            for attempt in range(max_attempts):
                try:
                    request_count += 1
                    
                    # Minimal delay before request
                    time.sleep(random.uniform(0.2, 0.5))
                    
                    # Match EXACT browser headers from working request
                    post_headers = {
                        'Accept': '*/*',
                        'Accept-Encoding': 'gzip, deflate, br, zstd',
                        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8,hi;q=0.7,gu;q=0.6',
                        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                        'Origin': BASE_URL.rstrip('/'),
                        'Referer': category_url,
                        'User-Agent': user_agent,  # Use consistent user agent
                        'X-Requested-With': 'XMLHttpRequest',
                        'sec-ch-ua': '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
                        'sec-ch-ua-mobile': '?0',
                        'sec-ch-ua-platform': '"Windows"',
                        'sec-fetch-dest': 'empty',
                        'sec-fetch-mode': 'cors',
                        'sec-fetch-site': 'same-origin',
                    }
                    
                    # Use session.post - session automatically includes cookies
                    r = session.post(
                        LOAD_MORE_URL, 
                        data=post_data,
                        headers=post_headers,
                        timeout=60
                    )
                    
                    # Check for 403
                    if r.status_code == 403:
                        print(f"   ⚠️  Attempt {attempt + 1}/{max_attempts}: 403 Forbidden")
                        
                        # If all attempts exhausted, stop
                        if attempt >= max_attempts - 1:
                            print(f"⚠️  403 Forbidden after {max_attempts} attempts - stopping load_more for this category")
                            consecutive_failures += 1
                            break
                        
                        # Try to refresh cookies IMMEDIATELY
                        print(f"   🔄 Attempting emergency cookie refresh...")
                        
                        # Fast exponential backoff
                        retry_wait = min(10, RETRY_DELAY_MIN * (2 ** attempt)) + random.uniform(0.5, 1)
                        print(f"   ⏳ Waiting {retry_wait:.1f}s before retry...")
                        time.sleep(retry_wait)
                        
                        # Try refreshing existing session first
                        if refresh_session_cookies(session, category_url, user_agent):
                            last_cookie_refresh = time.time()
                            time.sleep(random.uniform(0.5, 1))
                            continue  # Retry with refreshed cookies
                        
                        # If refresh failed, create completely new session with new token
                        print(f"   🔄 Cookie refresh failed, creating new session...")
                        session.close()
                        
                        new_session, new_user_agent, new_web_token = create_fresh_session(category_url)
                        if new_session:
                            session = new_session
                            user_agent = new_user_agent
                            if new_web_token:
                                web_token = new_web_token  # Update token if extracted
                                print(f"   🔑 Updated web_token: {web_token[:20]}...")
                            last_cookie_refresh = time.time()
                            time.sleep(random.uniform(0.5, 1))
                            continue  # Retry with new session
                        else:
                            print(f"   ⚠️  Failed to create new session")
                            consecutive_failures += 1
                            break
                    
                    r.raise_for_status()
                    
                    # SUCCESS - Update cookies from response automatically handled by session
                    # Parse response
                    soup = BeautifulSoup(r.content, 'html.parser')
                    divs = soup.find_all('div', class_=re.compile(r'col-xs-6'))
                    
                    if not divs or len(divs) == 0:
                        print(f"✅ No more products available. Total loaded: {len(all_divs)}")
                        loaded_in_this_iteration = False
                        break  # Exit loop - no more products
                    
                    all_divs.extend(divs)
                    print(f"   ✓ Loaded {len(divs)} more products (Total: {len(all_divs)})")
                    
                    offset += len(divs)
                    loaded_in_this_iteration = True
                    consecutive_failures = 0  # Reset failure counter on success
                    
                    # Apply limit if specified
                    if limit_per_category and len(all_divs) >= limit_per_category:
                        all_divs = all_divs[:limit_per_category]
                        print(f"📌 Reached limit of {limit_per_category} products")
                        loaded_in_this_iteration = False
                        break  # Exit loop - limit reached
                    
                    # Success, exit retry loop and continue to next batch
                    break
                    
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code == 403:
                            print(f"   ⚠️  Attempt {attempt + 1}/{max_attempts}: 403 Forbidden")
                            if attempt < max_attempts - 1:
                                retry_wait = random.uniform(2, 4) * (attempt + 1)
                                print(f"   ⏳ Waiting {retry_wait:.1f}s before retry...")
                                time.sleep(retry_wait)
                                continue
                        else:
                            print(f"❌ HTTP Error: {e}")
                    else:
                        print(f"❌ HTTP Error: {e}")
                    
                    if attempt == max_attempts - 1:
                        consecutive_failures += 1
                    break
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    if attempt == max_attempts - 1:
                        consecutive_failures += 1
                    break
            
            # Check if we should stop due to consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"⚠️  Stopping after {consecutive_failures} consecutive failures")
                print(f"   Total products loaded: {len(all_divs)}")
                break
            
            # If nothing loaded in this iteration and no retry left, stop
            if not loaded_in_this_iteration:
                break
        
        print(f"\n✅ Total products found in {category_name}: {len(all_divs)}")
        
        # Extract product info from all divs
        products = []
        for div in all_divs:
            try:
                link_tag = div.find('a', href=re.compile(r'watchvine01.cartpe.in/.*html'))
                if not link_tag:
                    continue
                p_url = link_tag['href']
                p_name = (div.find('h5') or div.find('h3')).get_text(strip=True)
                products.append((p_url, p_name, category_key, category_name))
            except Exception as e:
                continue
        
        session.close()
        return products
        
    except Exception as e:
        print(f"❌ Error scraping {category_name}: {e}")
        session.close()
        return []


def scrape_all_products(limit_per_category=None, clear_db=False, specific_categories=None):
    """Main scraping function with category-wise parallel processing.
    
    Args:
        limit_per_category: Max products per category (None = all)
        clear_db: Clear database before scraping
        specific_categories: List of category keys to scrape (None = all categories)
    """
    print("="*80)
    print("🚀 CATEGORY-WISE MULTI-THREADED SCRAPER")
    print("="*80)
    
    # Connect to MongoDB
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    products_col = db[COLLECTION_NAME]
    
    if clear_db:
        print("🗑️  Clearing existing data...")
        products_col.delete_many({})
        print("✅ Database cleared")
    
    # Create index on category_key for faster searches
    try:
        products_col.create_index([("category_key", 1)])
        products_col.create_index([("name", "text")])
        print("📑 Created database indexes for optimized searching")
    except:
        pass
    
    # Determine which categories to scrape
    categories_to_scrape = CATEGORIES
    if specific_categories:
        categories_to_scrape = {k: v for k, v in CATEGORIES.items() if k in specific_categories}
    
    print(f"\n📋 Categories to scrape: {len(categories_to_scrape)}")
    for key, info in categories_to_scrape.items():
        print(f"   • {info['name']}")
    
    # Scrape all categories
    all_products = []
    for cat_idx, (cat_key, cat_info) in enumerate(categories_to_scrape.items(), 1):
        print(f"\n🔄 Processing category {cat_idx}/{len(categories_to_scrape)}: {cat_info['name']}")
        products = scrape_category(cat_key, cat_info, limit_per_category)
        all_products.extend(products)
        
        # Short pause between categories
        if cat_idx < len(categories_to_scrape):
            wait_time = random.uniform(1, 2)
            print(f"⏸️  Waiting {wait_time:.1f}s before next category...")
            time.sleep(wait_time)
    
    print(f"\n✅ Total products found across all categories: {len(all_products)}")
    print(f"⚙️  Using {MAX_WORKERS} parallel threads")
    print(f"💾 Saving in batches of {BATCH_SIZE}")
    print(f"\n{'='*80}")
    print("Starting detailed scrape...\n")
    
    # Prepare product data with index
    products_to_scrape = []
    for idx, (p_url, p_name, cat_key, cat_name) in enumerate(all_products):
        products_to_scrape.append((p_url, p_name, cat_key, cat_name, idx + 1, len(all_products)))
    
    stats['total'] = len(products_to_scrape)
    stats['start_time'] = time.time()
    
    # Parallel scraping
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scrape_single_product, product) for product in products_to_scrape]
        
        # Wait for completion
        for future in as_completed(futures):
            pass
    
    # Save remaining products in buffer
    if products_buffer:
        save_batch_to_db(products_buffer)
        products_buffer.clear()
    
    # Final statistics with category breakdown
    elapsed = time.time() - stats['start_time']
    print(f"\n{'='*80}")
    print("📊 SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Successful: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"⏱️  Total time: {elapsed/60:.2f} minutes")
    print(f"⚡ Average rate: {stats['success']/elapsed:.2f} products/second")
    print(f"💾 Saved to: {DB_NAME}.{COLLECTION_NAME}")
    
    # Show category-wise stats
    print(f"\n📊 Category-wise Distribution:")
    for cat_key in categories_to_scrape.keys():
        count = products_col.count_documents({"category_key": cat_key})
        cat_name = CATEGORIES[cat_key]['name']
        print(f"   • {cat_name}: {count} products")
    
    client.close()


def main(limit_per_category=None, clear_db=False, workers=None, categories=None):
    """Main function with configurable parameters.
    
    Args:
        limit_per_category: Max products per category (None = all products)
        clear_db: Whether to clear existing MongoDB data (default: False)
        workers: Number of parallel threads (default: use MAX_WORKERS)
        categories: List of specific category keys to scrape (None = all categories)
    """
    global MAX_WORKERS
    
    if workers:
        MAX_WORKERS = workers
    
    print("\n" + "="*80)
    print("🔧 CATEGORY-WISE SCRAPER - Configuration")
    print("="*80)
    print(f"Products per category: {limit_per_category if limit_per_category else 'ALL'}")
    print(f"Clear database: {'Yes' if clear_db else 'No'}")
    print(f"Parallel threads: {MAX_WORKERS}")
    if categories:
        print(f"Specific categories: {', '.join(categories)}")
    else:
        print(f"Categories: ALL ({len(CATEGORIES)} categories)")
    print(f"\n⚠️  Press Ctrl+C at any time to stop and save progress\n")
    
    time.sleep(1)
    
    scrape_all_products(limit_per_category, clear_db, categories)


if __name__ == '__main__':
    import sys
    
    # Check if arguments provided
    if len(sys.argv) > 1:
        # Command line mode: python fast_scraper.py <limit_per_category> [clear_db] [workers] [categories...]
        try:
            limit = int(sys.argv[1]) if sys.argv[1].lower() != 'all' else None
            clear_db = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else False
            workers = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
            categories = sys.argv[4:] if len(sys.argv) > 4 else None
            main(limit_per_category=limit, clear_db=clear_db, workers=workers, categories=categories)
        except (ValueError, IndexError) as e:
            print("Usage: python fast_scraper.py <limit|all> [true|false] [workers] [category_keys...]")
            print("\nExamples:")
            print("  python fast_scraper.py 10 true 15                    # 10 products per category, clear DB, 15 threads")
            print("  python fast_scraper.py all false 10                  # All products, keep DB, 10 threads")
            print("  python fast_scraper.py 5 true 10 mens_watch wallet  # Only mens_watch and wallet categories")
            print("\nAvailable category keys:")
            for key, info in CATEGORIES.items():
                print(f"  - {key}: {info['name']}")
            sys.exit(1)
    else:
        # Interactive mode
        print("\n" + "="*80)
        print("🔧 CATEGORY-WISE SCRAPER - Interactive Mode")
        print("="*80)
        
        print("\nAvailable categories:")
        for idx, (key, info) in enumerate(CATEGORIES.items(), 1):
            print(f"  {idx}. {key}: {info['name']}")
        
        limit_input = input(f"\nEnter products per category (press Enter for ALL): ").strip()
        limit = int(limit_input) if limit_input else None
        
        clear_input = input("Clear existing data in MongoDB? (y/N): ").strip().lower()
        clear_db = clear_input == 'y'
        
        workers_input = input(f"Number of parallel threads (default {MAX_WORKERS}): ").strip()
        workers = int(workers_input) if workers_input else None
        
        categories_input = input("Specific categories (comma-separated keys, Enter for all): ").strip()
        categories = [c.strip() for c in categories_input.split(',')] if categories_input else None
        
        main(limit_per_category=limit, clear_db=clear_db, workers=workers, categories=categories)
