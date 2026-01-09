"""
Google Sheets Integration for Order Management
Saves customer orders to Google Sheets
"""

import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GoogleSheetsHandler:
    """Handle Google Sheets operations for order management"""
    
    def __init__(self, credentials_file: str = "credentials.json", sheet_url: str = None):
        """
        Initialize Google Sheets handler
        
        Args:
            credentials_file: Path to Google service account JSON
            sheet_url: Google Sheets URL from environment
        """
        self.credentials_file = credentials_file
        self.sheet_url = sheet_url or os.getenv("GOOGLE_SHEET_URL")
        self.client = None
        self.sheet = None
        self.worksheet = None
        
        # Only initialize if credentials exist
        if os.path.exists(credentials_file):
            self._initialize_client()
        else:
            logger.warning(f"Credentials file not found: {credentials_file}")
    
    def _initialize_client(self):
        """Initialize Google Sheets client"""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_file, scope
            )
            self.client = gspread.authorize(creds)
            
            if self.sheet_url:
                self.sheet = self.client.open_by_url(self.sheet_url)
                self.worksheet = self.sheet.sheet1  # First sheet
                logger.info("✅ Google Sheets connected successfully")
            else:
                logger.warning("No Google Sheet URL provided")
                
        except Exception as e:
            logger.error(f"Error initializing Google Sheets: {e}")
    
    def save_order(self, order_data: dict) -> bool:
        """
        Save order to Google Sheets
        
        Args:
            order_data: Dictionary containing order information
                - customer_name
                - phone_number
                - email
                - address
                - product_name
                - product_url
                - quantity
                - timestamp
        
        Returns:
            bool: True if saved successfully
        """
        try:
            if not self.worksheet:
                logger.error("❌ Google Sheets not initialized - worksheet is None")
                logger.error(f"   - Client: {self.client is not None}")
                logger.error(f"   - Sheet URL: {self.sheet_url}")
                logger.error(f"   - Credentials file exists: {os.path.exists(self.credentials_file)}")
                return False
            
            logger.info(f"📝 Preparing to save order to Google Sheets: {order_data.get('order_id')}")
            
            # Prepare row data
            row = [
                order_data.get('order_id', ''),
                order_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                order_data.get('customer_name', ''),
                order_data.get('phone_number', ''),
                order_data.get('email', ''),
                order_data.get('address', ''),
                order_data.get('product_name', ''),
                order_data.get('product_url', ''),
                order_data.get('quantity', 1),
                order_data.get('status', 'Pending'),
                order_data.get('notes', '')
            ]
            
            logger.info(f"📊 Row data prepared: {len(row)} columns")
            
            # Append to sheet
            self.worksheet.append_row(row)
            logger.info(f"✅ Order saved to Google Sheets successfully: {order_data.get('order_id')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving order to Google Sheets: {e}")
            logger.exception("Full traceback:")
            return False
    
    def initialize_sheet_headers(self):
        """Initialize sheet with headers if empty"""
        try:
            if not self.worksheet:
                return False
            
            # Check if sheet is empty
            if len(self.worksheet.get_all_values()) == 0:
                headers = [
                    'Order ID',
                    'Timestamp',
                    'Customer Name',
                    'Phone Number',
                    'Email',
                    'Address',
                    'Product Name',
                    'Product URL',
                    'Quantity',
                    'Status',
                    'Notes'
                ]
                self.worksheet.append_row(headers)
                logger.info("✅ Sheet headers initialized")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing headers: {e}")
            return False
    
    def get_order_count(self) -> int:
        """Get total number of orders"""
        try:
            if not self.worksheet:
                return 0
            
            all_values = self.worksheet.get_all_values()
            # Subtract 1 for header row
            return max(0, len(all_values) - 1)
            
        except Exception as e:
            logger.error(f"Error getting order count: {e}")
            return 0


# Fallback: Save to MongoDB if Google Sheets not available
class MongoOrderStorage:
    """Fallback storage using MongoDB"""
    
    def __init__(self, mongodb_uri: str = None, db_name: str = None):
        from pymongo import MongoClient
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = db_name or os.getenv("MONGODB_DB", "whatsapp_bot")
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.db_name]
        self.orders = self.db.orders
        self.orders.create_index("order_id", unique=True)
        logger.info(f"✅ MongoDB order storage initialized")
    
    def save_order(self, order_data: dict) -> bool:
        """Save order to MongoDB"""
        try:
            order_doc = {
                'order_id': order_data.get('order_id', ''),
                'timestamp': order_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'customer_name': order_data.get('customer_name', ''),
                'phone_number': order_data.get('phone_number', ''),
                'email': order_data.get('email', ''),
                'address': order_data.get('address', ''),
                'product_name': order_data.get('product_name', ''),
                'product_url': order_data.get('product_url', ''),
                'quantity': order_data.get('quantity', 1),
                'status': order_data.get('status', 'Pending'),
                'notes': order_data.get('notes', '')
            }
            self.orders.insert_one(order_doc)
            logger.info(f"✅ Order saved to MongoDB: {order_data.get('order_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving order to MongoDB: {e}")
            return False
