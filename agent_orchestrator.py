"""
Multi-Agent Orchestrator
Decides which agent/function to call based on conversation state
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation states"""
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    BROWSING = "browsing"
    PRODUCT_SELECTED = "product_selected"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COLLECTING_DETAILS = "collecting_details"
    DETAILS_COLLECTED = "details_collected"
    AWAITING_FINAL_CONFIRMATION = "awaiting_final_confirmation"
    ORDER_PLACED = "order_placed"
    COMPLETED = "completed"


class OrderData:
    """Order data structure"""
    def __init__(self):
        self.customer_name: Optional[str] = None
        self.phone_number: Optional[str] = None
        self.email: Optional[str] = None
        self.address: Optional[str] = None
        self.product_name: Optional[str] = None
        self.product_url: Optional[str] = None
        self.quantity: int = 1
        self.order_id: Optional[str] = None
        self.timestamp: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        return all([
            self.customer_name,
            self.phone_number,
            self.address,
            self.product_name or self.product_url
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'timestamp': self.timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'customer_name': self.customer_name,
            'phone_number': self.phone_number,
            'email': self.email or 'N/A',
            'address': self.address,
            'product_name': self.product_name or 'N/A',
            'product_url': self.product_url or 'N/A',
            'quantity': self.quantity,
            'status': 'Pending',
            'notes': ''
        }


class AgentOrchestrator:
    """
    Orchestrator that decides which action to take based on conversation state
    """
    
    def __init__(self, conversation_manager):
        self.conversation_manager = conversation_manager
        self.user_states: Dict[str, ConversationState] = {}
        self.user_orders: Dict[str, OrderData] = {}
        self.cached_products: Dict[str, list] = {}  # Cache product data for "more images" requests
    
    def get_user_state(self, phone_number: str) -> ConversationState:
        """Get current conversation state for user"""
        return self.user_states.get(phone_number, ConversationState.GREETING)
    
    def set_user_state(self, phone_number: str, state: ConversationState):
        """Set conversation state for user"""
        self.user_states[phone_number] = state
        logger.info(f"📊 State updated for {phone_number}: {state.value}")
    
    def get_order_data(self, phone_number: str) -> OrderData:
        """Get order data for user"""
        if phone_number not in self.user_orders:
            self.user_orders[phone_number] = OrderData()
        return self.user_orders[phone_number]
    
    def analyze_message(self, message: str, phone_number: str) -> Tuple[str, dict]:
        """
        Analyze user message and decide action using Backend Tool Classifier
        
        Returns:
            Tuple of (action, metadata)
            Actions: 'ai_response', 'save_order_direct'
        """
        logger.info(f"📨 Analyzing message from {phone_number}")
        
        # Get conversation history
        conversation_history = self.conversation_manager.get_history(phone_number)
        
        # Get search context for pagination (if exists)
        search_context = self.get_search_context(phone_number)
        
        # Use backend tool classifier with Gemini
        from backend_tool_classifier import BackendToolClassifier
        classifier = BackendToolClassifier()
        
        # Classify the message (pass search context for pagination)
        decision = classifier.analyze_and_classify(conversation_history, message, phone_number, search_context)
        
        tool = decision.get('tool', 'ai_chat')
        
        logger.info(f"🔧 Backend Tool Classifier Decision: {tool}")
        
        if tool == 'find_product':
            # Backend AI detected product search and extracted keyword + range
            keyword = decision.get('keyword', '')
            range_str = decision.get('range', '0-10')
            logger.info(f"🔍 Backend AI extracted product keyword: '{keyword}'")
            logger.info(f"📊 Backend AI provided range: '{range_str}'")
            return ('find_product', {'keyword': keyword, 'range': range_str})
        
        elif tool == 'ask_product_for_images':
            # User wants more images but didn't specify which product
            logger.info(f"📸 Backend AI detected: User wants more images (no product specified)")
            return ('ask_product_for_images', {})
        
        elif tool == 'send_all_images':
            # User wants all images for specific product
            product_name = decision.get('product_name', '')
            logger.info(f"📸 Backend AI detected: User wants all images for '{product_name}'")
            return ('send_all_images', {'product_name': product_name})
        
        elif tool == 'show_all_cached_images':
            # User wants to see all images from recent search
            logger.info(f"📸 Backend AI detected: User wants all images for all recent products")
            return ('show_all_cached_images', {})
        
        elif tool == 'show_category_products':
            # User wants to see products from a category
            logger.info(f"📂 Backend AI detected: User wants to see all products from category")
            return ('show_category_products', {})
        
        elif tool == 'save_data_to_google_sheet':
            # Backend AI decided to save order directly
            order_data = decision.get('data', {})
            logger.info(f"💾 Backend AI extracted order data: {order_data}")
            return ('save_order_direct', {'order_data': order_data})
        
        else:
            # Default: Let chat AI respond
            return ('ai_response', {'intent': 'general_query'})
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        import re
        greetings = ['hello', 'hi', 'hey', 'namaste', 'good morning', 
                    'good afternoon', 'good evening', 'hola']
        # Use word boundaries to avoid matching 'hi' in 'this' or 'hey' in 'they'
        for greeting in greetings:
            # Check if greeting appears as a whole word
            if re.search(r'\b' + re.escape(greeting) + r'\b', message, re.IGNORECASE):
                return True
        return False
    
    def _is_product_url(self, message: str) -> bool:
        """Check if message contains product URL"""
        return 'watchvine01.cartpe.in' in message.lower() or 'http' in message.lower()
    
    def _extract_product_url(self, message: str) -> str:
        """Extract product URL from message"""
        import re
        url_pattern = r'(https?://[^\s]+)'
        match = re.search(url_pattern, message)
        return match.group(1) if match else ''
    
    def _extract_product_name(self, message: str) -> str:
        """Extract product name from message (sent via WhatsApp button)"""
        # Product name usually comes before the URL
        parts = message.split('http')
        if len(parts) > 1:
            # Get the text before the URL
            text_before_url = parts[0].strip()
            
            # Remove common prefixes (case-insensitive)
            # Order matters - check longer phrases first!
            prefixes_to_remove = [
                'i want to buy this',
                'I want to buy this',
                'i am interested in',
                'I am interested in',
                'i want to buy',
                'I want to buy',
                'check out',
                'Check out',
                'buy',
                'Buy'
            ]
            
            # Try to remove any matching prefix (case-insensitive)
            name = text_before_url
            for prefix in prefixes_to_remove:
                if text_before_url.lower().startswith(prefix.lower()):
                    # Remove the prefix
                    name = text_before_url[len(prefix):].strip()
                    break
            
            # If we have a valid product name after removing prefix, return it
            if name and len(name) > 2 and name.lower() != text_before_url.lower():
                return name
        
        # Fallback: Try to extract product name from URL
        # e.g., https://watchvine01.cartpe.in/products/rolex-watch
        url = self._extract_product_url(message)
        if url:
            # Extract from URL path
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                path_parts = parsed.path.strip('/').split('/')
                if len(path_parts) > 0:
                    # Get last part of path and clean it up
                    product_slug = path_parts[-1]
                    # Remove .html, .php, etc.
                    product_slug = product_slug.split('.')[0]
                    # Convert slug to readable name (e.g., rolex-watch -> Rolex Watch)
                    name = product_slug.replace('-', ' ').replace('_', ' ').title()
                    if name and len(name) > 2:
                        return name
            except Exception as e:
                logger.warning(f"Could not extract product name from URL: {e}")
        
        return ''
    
    def _is_confirmation(self, message: str) -> bool:
        """Check if message is a confirmation"""
        confirmations = ['yes', 'confirm', 'conform', 'ha', 'haan', 'ok', 'okay', 
                        'sure', 'proceed', 'continue', 'yes please', 'correct', 
                        'right', 'yup', 'yeah', 'yep']
        return any(conf in message for conf in confirmations)
    
    def _extract_order_details(self, message: str, order_data: OrderData, phone_number: str) -> bool:
        """
        Extract order details from message
        Returns True if new details were extracted
        """
        import re
        
        # Handle both markdown format and plain text
        lines = message.split('\n')
        
        for line in lines:
            # Remove markdown formatting (*, **, bullets, etc.)
            line_clean = line.replace('*', '').replace('-', '').strip()
            line_lower = line_clean.lower()
            
            # Skip empty lines and headers
            if not line_clean or 'order details' in line_lower:
                continue
            
            # Extract name (look for "name:" or "customer name:")
            if not order_data.customer_name and 'name' in line_lower and ':' in line_clean:
                name = line_clean.split(':')[-1].strip()
                # Validate: name should be 2+ chars and not contain too many numbers
                if len(name) > 2 and sum(c.isdigit() for c in name) < len(name) / 2:
                    order_data.customer_name = name
                    logger.info(f"📝 Extracted name: {name}")
            
            # Extract phone (look for "phone:" or numbers)
            if not order_data.phone_number and ('phone' in line_lower or 'mobile' in line_lower or 'number' in line_lower):
                phone_match = re.search(r'[\d\s\-\+]{10,}', line_clean)
                if phone_match:
                    order_data.phone_number = phone_match.group().strip()
                    logger.info(f"📝 Extracted phone: {order_data.phone_number}")
            
            # Extract email
            if not order_data.email and '@' in line_clean:
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', line_clean)
                if email_match:
                    order_data.email = email_match.group()
                    logger.info(f"📝 Extracted email: {order_data.email}")
            
            # Extract address (look for "address:" or long text with address keywords)
            if not order_data.address and 'address' in line_lower and ':' in line_clean:
                address = line_clean.split(':')[-1].strip()
                if len(address) > 10:  # Address should be reasonably long
                    order_data.address = address
                    logger.info(f"📝 Extracted address: {address[:50]}...")
            
            # Extract product name
            if not order_data.product_name and 'product' in line_lower and 'name' in line_lower and ':' in line_clean:
                product = line_clean.split(':')[-1].strip()
                if len(product) > 2:
                    order_data.product_name = product
                    logger.info(f"📝 Extracted product name: {product}")
            
            # Extract quantity
            if 'quantity' in line_lower and ':' in line_clean:
                qty_match = re.search(r'\d+', line_clean.split(':')[-1])
                if qty_match:
                    order_data.quantity = int(qty_match.group())
                    logger.info(f"📝 Extracted quantity: {order_data.quantity}")
        
        # Fallback: if phone_number not in order, use the one from WhatsApp
        if not order_data.phone_number:
            order_data.phone_number = phone_number
            logger.info(f"📝 Using WhatsApp phone: {phone_number}")
        
        # Log current status
        logger.info(f"📊 Order data status - Name: {bool(order_data.customer_name)}, "
                   f"Phone: {bool(order_data.phone_number)}, "
                   f"Address: {bool(order_data.address)}, "
                   f"Complete: {order_data.is_complete()}")
        
        return True
    
    def _generate_order_id(self, phone_number: str) -> str:
        """Generate unique order ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        phone_suffix = phone_number[-4:] if len(phone_number) >= 4 else phone_number
        return f"WV{timestamp}{phone_suffix}"
    
    def get_search_context(self, phone_number: str) -> dict:
        """
        Get search context for pagination
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Dict with keyword, total_found, sent_count (or empty dict if no context)
        """
        try:
            context_doc = self.conversation_manager.conversations.find_one(
                {'phone_number': phone_number, 'role': 'search_context'}
            )
            
            if context_doc:
                return {
                    'keyword': context_doc.get('keyword', ''),
                    'total_found': context_doc.get('total_found', 0),
                    'sent_count': context_doc.get('sent_count', 0)
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting search context: {e}")
            return {}
    
    def save_search_context(self, phone_number: str, keyword: str, total_found: int, sent_count: int):
        """
        Save search context for pagination (show more functionality)
        
        Args:
            phone_number: User's phone number
            keyword: Search keyword used
            total_found: Total products found
            sent_count: Number of products already sent
        """
        try:
            context = {
                'keyword': keyword,
                'total_found': total_found,
                'sent_count': sent_count,
                'timestamp': datetime.now()
            }
            self.conversation_manager.conversations.update_one(
                {'phone_number': phone_number, 'role': 'search_context'},
                {'$set': context},
                upsert=True
            )
            logger.info(f"💾 Search context saved: {keyword} ({sent_count}/{total_found})")
        except Exception as e:
            logger.error(f"Error saving search context: {e}")
    
    def cache_product_data(self, phone_number: str, products: list):
        """Cache product data for future 'more images' requests - Persistent MongoDB storage"""
        try:
            from datetime import datetime, timedelta
            
            # Store in MongoDB for persistence across restarts
            self.conversation_manager.db.product_cache.update_one(
                {'phone_number': phone_number},
                {
                    '$set': {
                        'phone_number': phone_number,
                        'products': products,
                        'sent_count': 0,  # Track how many products already sent for pagination
                        'timestamp': datetime.now(),
                        'expires_at': datetime.now() + timedelta(hours=24)  # Cache for 24 hours
                    }
                },
                upsert=True
            )
            
            # Also keep in memory for faster access
            self.cached_products[phone_number] = products
            logger.info(f"💾 Cached {len(products)} products for {phone_number} (MongoDB + Memory)")
        except Exception as e:
            logger.error(f"Error caching products to MongoDB: {e}")
            # Fallback to memory-only cache
            self.cached_products[phone_number] = products
            logger.info(f"💾 Cached {len(products)} products for {phone_number} (Memory only)")
    
    def get_cached_products(self, phone_number: str) -> list:
        """Get cached product data - Check memory first, then MongoDB"""
        # First check in-memory cache
        if phone_number in self.cached_products:
            logger.info(f"📦 Found {len(self.cached_products[phone_number])} products in memory cache")
            return self.cached_products.get(phone_number, [])
        
        # If not in memory, check MongoDB
        try:
            from datetime import datetime
            
            cached_doc = self.conversation_manager.db.product_cache.find_one(
                {
                    'phone_number': phone_number,
                    'expires_at': {'$gt': datetime.now()}  # Not expired
                }
            )
            
            if cached_doc and cached_doc.get('products'):
                products = cached_doc['products']
                # Restore to memory cache for faster future access
                self.cached_products[phone_number] = products
                logger.info(f"📦 Found {len(products)} products in MongoDB cache")
                return products
            else:
                logger.info(f"📦 No cached products found in MongoDB for {phone_number}")
                return []
        except Exception as e:
            logger.error(f"Error retrieving cached products from MongoDB: {e}")
            return []
    
    def get_next_cached_products(self, phone_number: str, batch_size: int = 5) -> tuple:
        """
        Get next batch of cached products for pagination (show more functionality)
        
        Args:
            phone_number: User's phone number
            batch_size: Number of products to return in this batch
        
        Returns:
            Tuple of (next_batch_products, has_more, sent_count, total_count)
        """
        try:
            from datetime import datetime
            
            # Get cached data from MongoDB
            cached_doc = self.conversation_manager.db.product_cache.find_one(
                {
                    'phone_number': phone_number,
                    'expires_at': {'$gt': datetime.now()}  # Not expired
                }
            )
            
            if not cached_doc or not cached_doc.get('products'):
                logger.warning(f"⚠️ No cached products for pagination: {phone_number}")
                return ([], False, 0, 0)
            
            products = cached_doc['products']
            sent_count = cached_doc.get('sent_count', 0)
            total_count = len(products)
            
            # Calculate next batch
            start_idx = sent_count
            end_idx = min(sent_count + batch_size, total_count)
            next_batch = products[start_idx:end_idx]
            
            # Update sent_count in MongoDB
            new_sent_count = end_idx
            self.conversation_manager.db.product_cache.update_one(
                {'phone_number': phone_number},
                {'$set': {'sent_count': new_sent_count}}
            )
            
            has_more = new_sent_count < total_count
            
            logger.info(f"📄 Pagination: Sending products {start_idx+1}-{end_idx} of {total_count} (has_more={has_more})")
            
            return (next_batch, has_more, new_sent_count, total_count)
            
        except Exception as e:
            logger.error(f"Error getting next cached products: {e}")
            return ([], False, 0, 0)
    
    def set_user_context(self, phone_number: str, context: dict):
        """Set temporary context for user (e.g., last category searched)"""
        if phone_number not in self.user_states:
            self.user_states[phone_number] = {}
        # Store context in user_states as a dict
        if not isinstance(self.user_states[phone_number], dict):
            self.user_states[phone_number] = {'state': self.user_states[phone_number]}
        self.user_states[phone_number]['context'] = context
        logger.info(f"💾 Saved context for {phone_number}: {context}")
    
    def get_user_context(self, phone_number: str) -> dict:
        """Get temporary context for user"""
        if phone_number in self.user_states:
            if isinstance(self.user_states[phone_number], dict):
                return self.user_states[phone_number].get('context', {})
        return {}
    
    def clear_user_data(self, phone_number: str):
        """Clear user data after order completion"""
        if phone_number in self.user_states:
            del self.user_states[phone_number]
        if phone_number in self.user_orders:
            del self.user_orders[phone_number]
        if phone_number in self.cached_products:
            del self.cached_products[phone_number]
        
        # Also clear from MongoDB
        try:
            self.conversation_manager.db.product_cache.delete_one({'phone_number': phone_number})
            logger.info(f"🧹 Cleared data for {phone_number} (Memory + MongoDB)")
        except Exception as e:
            logger.error(f"Error clearing MongoDB cache: {e}")
            logger.info(f"🧹 Cleared data for {phone_number} (Memory only)")
