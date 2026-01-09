"""
Google Apps Script Handler for Order Management
Sends order data to Google Apps Script Web App instead of direct Google Sheets API
"""

import os
import requests
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GoogleAppsScriptHandler:
    """Handle order saving via Google Apps Script Web App"""
    
    def __init__(self, web_app_url: str = None, secret_key: str = None):
        """
        Initialize Google Apps Script handler
        
        Args:
            web_app_url: Google Apps Script Web App URL
            secret_key: Secret key for authentication
        """
        self.web_app_url = web_app_url or os.getenv("GOOGLE_APPS_SCRIPT_URL")
        self.secret_key = secret_key or os.getenv("GOOGLE_APPS_SCRIPT_SECRET")
        self.is_initialized = False
        
        if self.web_app_url and self.secret_key:
            self.is_initialized = True
            logger.info("✅ Google Apps Script handler initialized")
            logger.info(f"   Web App URL: {self.web_app_url[:50]}...")
        else:
            if not self.web_app_url:
                logger.warning("⚠️  GOOGLE_APPS_SCRIPT_URL not set in .env")
            if not self.secret_key:
                logger.warning("⚠️  GOOGLE_APPS_SCRIPT_SECRET not set in .env")
    
    def save_order(self, order_data: Dict) -> bool:
        """
        Save order to Google Sheets via Apps Script
        
        Args:
            order_data: Dictionary containing order information
                - order_id
                - timestamp
                - customer_name
                - phone_number
                - email
                - address
                - product_name
                - product_url
                - quantity
                - status
                - notes
        
        Returns:
            bool: True if saved successfully
        """
        try:
            if not self.is_initialized:
                logger.error("❌❌❌ Google Apps Script NOT INITIALIZED ❌❌❌")
                logger.error("   Missing GOOGLE_APPS_SCRIPT_URL or GOOGLE_APPS_SCRIPT_SECRET in .env")
                logger.error(f"   web_app_url present: {self.web_app_url is not None}")
                logger.error(f"   secret_key present: {self.secret_key is not None}")
                return False
            
            logger.info(f"📝 GOOGLE APPS SCRIPT - Preparing to save order: {order_data.get('order_id')}")
            logger.info(f"   🌐 Apps Script URL: {self.web_app_url[:50]}...")
            logger.info(f"   🔐 Secret key present: {bool(self.secret_key)}")
            
            # Prepare payload
            payload = {
                "secret": self.secret_key,
                "order": {
                    "order_id": order_data.get('order_id', ''),
                    "timestamp": order_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    "customer_name": order_data.get('customer_name', ''),
                    "phone_number": order_data.get('phone_number', ''),
                    "email": order_data.get('email', 'N/A'),
                    "address": order_data.get('address', ''),
                    "product_name": order_data.get('product_name', ''),
                    "product_url": order_data.get('product_url', ''),
                    "quantity": order_data.get('quantity', 1),
                    "status": order_data.get('status', 'Pending'),
                    "notes": order_data.get('notes', '')
                }
            }
            
            logger.info(f"📦 Payload prepared with {len(payload['order'])} fields")
            logger.info(f"   Order ID: {payload['order']['order_id']}")
            logger.info(f"   Customer: {payload['order']['customer_name']}")
            logger.info(f"   Phone: {payload['order']['phone_number']}")
            logger.info(f"   Product: {payload['order']['product_name']}")
            
            logger.info(f"📤 Sending POST request to Apps Script...")
            
            # Send POST request to Apps Script
            response = requests.post(
                self.web_app_url,
                json=payload,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            logger.info(f"📥 Response received - Status: {response.status_code}")
            logger.info(f"📄 Response body: {response.text[:500]}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"📋 Parsed JSON response: {result}")
                
                if result.get('success'):
                    logger.info(f"✅✅✅ ORDER SAVED VIA APPS SCRIPT: {order_data.get('order_id')} ✅✅✅")
                    logger.info(f"   Response message: {result.get('message')}")
                    return True
                else:
                    logger.error(f"❌❌❌ APPS SCRIPT RETURNED ERROR ❌❌❌")
                    logger.error(f"   Error message: {result.get('message')}")
                    logger.error(f"   Full response: {result}")
                    return False
            else:
                logger.error(f"❌❌❌ HTTP ERROR {response.status_code} ❌❌❌")
                logger.error(f"   Response text: {response.text[:500]}")
                return False
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ TIMEOUT ERROR - Apps Script took too long (>30s) to respond")
            logger.error(f"   URL: {self.web_app_url}")
            return False
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ CONNECTION ERROR - Could not reach Apps Script URL")
            logger.error(f"   URL: {self.web_app_url}")
            logger.error(f"   Error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌❌❌ UNEXPECTED ERROR in save_order() ❌❌❌")
            logger.error(f"   Error: {e}")
            logger.exception("Full traceback:")
            return False
    
    def test_connection(self) -> bool:
        """
        Test connection to Google Apps Script Web App
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.is_initialized:
                logger.error("❌ Cannot test - Apps Script not initialized")
                return False
            
            logger.info("🧪 Testing connection to Google Apps Script...")
            
            # Send GET request (for testing)
            response = requests.get(self.web_app_url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Connection successful!")
                logger.info(f"   Message: {result.get('message')}")
                return True
            else:
                logger.error(f"❌ Connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def get_order_count(self) -> int:
        """
        Get total number of orders (not implemented for Apps Script)
        This would require additional endpoint in Apps Script
        
        Returns:
            int: 0 (placeholder)
        """
        logger.warning("⚠️  get_order_count not implemented for Apps Script handler")
        return 0


# Standalone test function
def test_google_apps_script():
    """Test Google Apps Script integration"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "=" * 60)
    print("TESTING GOOGLE APPS SCRIPT INTEGRATION")
    print("=" * 60)
    
    # Check environment variables
    web_app_url = os.getenv("GOOGLE_APPS_SCRIPT_URL")
    secret_key = os.getenv("GOOGLE_APPS_SCRIPT_SECRET")
    
    print(f"\n1. GOOGLE_APPS_SCRIPT_URL set: {web_app_url is not None}")
    if web_app_url:
        print(f"   URL: {web_app_url[:60]}...")
    
    print(f"2. GOOGLE_APPS_SCRIPT_SECRET set: {secret_key is not None}")
    if secret_key:
        print(f"   Secret: {secret_key[:10]}..." if len(secret_key) > 10 else "   Secret: (too short)")
    
    if not web_app_url or not secret_key:
        print("\n❌ ERROR: Missing configuration in .env file")
        print("   Add these to your .env:")
        print("   GOOGLE_APPS_SCRIPT_URL=your_web_app_url_here")
        print("   GOOGLE_APPS_SCRIPT_SECRET=your_secret_key_here")
        return False
    
    # Initialize handler
    print("\n3. Initializing handler...")
    handler = GoogleAppsScriptHandler(web_app_url, secret_key)
    
    if not handler.is_initialized:
        print("❌ Failed to initialize handler")
        return False
    
    # Test connection
    print("\n4. Testing connection...")
    if not handler.test_connection():
        print("❌ Connection test failed")
        return False
    
    # Test order save
    print("\n5. Testing order save...")
    test_order = {
        'order_id': f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'customer_name': 'Test Customer',
        'phone_number': '9999999999',
        'email': 'test@example.com',
        'address': 'Test Address, Test City, PIN 123456',
        'product_name': 'Test Product - Sample Watch',
        'product_url': 'https://watchvine01.cartpe.in/test-product',
        'quantity': 1,
        'status': 'Pending',
        'notes': 'Automated test order'
    }
    
    success = handler.save_order(test_order)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS! Test order saved successfully")
        print("   Check your Google Sheet to verify")
    else:
        print("❌ FAILED! Could not save test order")
        print("   Check the error messages above")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    test_google_apps_script()
