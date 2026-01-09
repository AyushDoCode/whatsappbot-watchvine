"""
WatchVine WhatsApp Bot - Multi-Agent Architecture
Main entry point with orchestrator for intelligent action handling
"""

import os
import logging
import time
from flask import Flask, request, jsonify
from datetime import datetime
from typing import Dict
from pymongo import MongoClient
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import caching

# Import custom modules
from system_prompt_config import get_system_prompt
from tool_calling_config import get_tool_calling_system_prompt
from agent_orchestrator import AgentOrchestrator, ConversationState
from google_sheets_handler import GoogleSheetsHandler, MongoOrderStorage
from google_apps_script_handler import GoogleAppsScriptHandler
from backend_tool_classifier import BackendToolClassifier

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Gemini API Configuration
GOOGLE_API_KEY = os.getenv("Google_api")
GOOGLE_MODEL = os.getenv("google_model", "gemini-1.5-flash-001")

if not GOOGLE_API_KEY:
    logger.warning("⚠️ Google_api not set! AI features may fail.")
else:
    logger.info(f"✅ Using Google Model: {GOOGLE_MODEL}")

# Other configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "whatsapp_bot")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "shop-bot")
GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL")
STORE_CONTACT_NUMBER = "9016220667" 


# ============================================================================
# MONGODB CONVERSATION MANAGER
# ============================================================================

class ConversationManager:
    """Manage user conversations in MongoDB"""
    
    def __init__(self, mongodb_uri: str, db_name: str):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.conversations = self.db.conversations
        self.conversations.create_index("phone_number")
        
        # Create indexes for product cache
        self.db.product_cache.create_index("phone_number")
        self.db.product_cache.create_index("expires_at")
        
        logger.info("✅ MongoDB connected with product cache support")
    
    def get_conversation(self, phone_number: str, limit: int = 10):
        """Get conversation history"""
        try:
            messages = list(
                self.conversations.find(
                    {"phone_number": phone_number}
                ).sort("timestamp", -1).limit(limit)
            )
            messages.reverse()
            return [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
                if msg.get("content")  # Only include messages with content
            ]
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return []
    
    def get_history(self, phone_number: str, limit: int = 10):
        """Alias for get_conversation - used by backend tool classifier"""
        return self.get_conversation(phone_number, limit)
    
    def save_message(self, phone_number: str, role: str, content: str):
        """Save message to history"""
        try:
            self.conversations.insert_one({
                "phone_number": phone_number,
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            })
        except Exception as e:
            logger.error(f"Error saving message: {e}")
    
    def clear_conversation(self, phone_number: str) -> int:
        """Clear conversation history"""
        try:
            result = self.conversations.delete_many({"phone_number": phone_number})
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return 0

# ============================================================================
# CONVERSATION AI AGENT
# ============================================================================

class ConversationAgent:
    """AI Agent for handling customer conversations with tool calling support using Gemini"""
    
    def __init__(self, api_key=None, model: str = None, system_prompt: str = None,
                 conversation_manager: ConversationManager = None, order_storage=None,
                 api_key_rotator=None):
        """
        Initialize Conversation Agent with Gemini
        """
        self.api_key = api_key or os.getenv("Google_api")
        
        # Set model name
        env_model = model or os.getenv("google_model", "gemini-1.5-flash-001")
        if not env_model.startswith("models/") and not env_model.startswith("gemini-"):
             self.model_name = f"models/{env_model}"
        else:
             self.model_name = env_model
             
        self.system_prompt = system_prompt
        self.conversation_manager = conversation_manager
        self.order_storage = order_storage
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Cache management
        self.cache_name = "watchvine_agent_system_prompt_v1"
        self.cached_content = None
        self.last_cache_update = 0
        self.CACHE_TTL = 3600 # 1 hour
        
        logger.info(f"✅ Conversation Agent using Gemini API")
        logger.info(f"✅ Conversation Agent initialized with model: {self.model_name}")

    def _get_or_create_cache(self):
        """Creates or retrieves cached content for system instructions"""
        if not self.api_key: return None
        
        current_time = time.time()
        if self.cached_content and (current_time - self.last_cache_update < self.CACHE_TTL):
            return self.cached_content

        try:
            # Check for existing cache (simplified: just create new one with TTL)
            # In production, you'd list and find by name to avoid duplication
            # For this simplified version, we'll try to create it.
            
            logger.info("🆕 Creating/Updating Agent Context Cache...")
            self.cached_content = caching.CachedContent.create(
                model=self.model_name,
                display_name=self.cache_name,
                system_instruction=self.system_prompt,
                ttl=datetime.timedelta(hours=2)
            )
            self.last_cache_update = current_time
            logger.info(f"✅ Agent Cache created: {self.cached_content.name}")
            return self.cached_content
        except Exception as e:
            logger.error(f"❌ Cache creation failed: {e}")
            return None

    def get_response(self, user_message: str, phone_number: str, 
                    metadata: dict = None) -> str:
        """Get AI response based on user message and context"""
        try:
            # SECURITY CHECK: Detect troll attempts
            troll_keywords = ['ignore', 'forget', 'override', 'act as', 'pretend', 'role-play', 'jailbreak']
            if any(keyword in user_message.lower() for keyword in troll_keywords):
                logger.warning(f"🚨 Troll attempt detected from {phone_number}")
                return "I'm WatchVine Assistant, here to help with your shopping needs. How may I assist you today? 😊"
            
            # Get conversation history
            history = self.conversation_manager.get_conversation(phone_number, limit=10)
            
            # Build context based on metadata
            context = self._build_context(user_message, metadata or {})
            
            # Construct chat history for Gemini
            chat_history = []
            for msg in history:
                role = "user" if msg['role'] == "user" else "model"
                chat_history.append({"role": role, "parts": [msg['content']]})
            
            # Try to use cache
            cached_content = self._get_or_create_cache()
            
            if cached_content:
                model = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
            else:
                model = genai.GenerativeModel(self.model_name, system_instruction=self.system_prompt)
            
            # Start chat session
            chat = model.start_chat(history=chat_history)
            
            # Send message with context
            full_message = f"{context}\n\nUser Message: {user_message}" if context else user_message
            
            response = chat.send_message(full_message, generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                max_output_tokens=450
            ))
            
            ai_response = response.text
            
            # Save conversation
            self.conversation_manager.save_message(phone_number, "user", user_message)
            self.conversation_manager.save_message(phone_number, "assistant", ai_response)
            
            logger.info(f"✅ Response generated for {phone_number}")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    def _build_context(self, user_message: str, metadata: dict) -> str:
        """Build context for AI based on intent and state"""
        context = ""
        intent = metadata.get('intent', 'general_query')
        
        # Check for color requests
        color_keywords = ['blue', 'red', 'black', 'white', 'pink', 'green', 'yellow', 'brown', 'purple', 'orange']
        if any(color in user_message.lower() for color in color_keywords):
            context += "\n\n⚠️ **IMPORTANT:** Customer mentioned a color. DO NOT claim we have that specific color unless confirmed in data. Provide browse URL instead.\n"
        
        # Intent-specific context
        if intent == 'greeting':
            context += "\n\n**CONTEXT:** This is the customer's first message. Greet them warmly as per system prompt.\n"
        
        elif intent == 'collect_details':
            context += "\n\n**CONTEXT:** Customer has selected a product. Now collect their details (Name, Phone, Address, Email if available, Quantity). Be friendly and guide them.\n"
        
        elif intent == 'missing_details':
            missing = metadata.get('missing_fields', [])
            context += f"\n\n**CONTEXT:** Customer tried to confirm but missing details: {', '.join(missing)}. Politely ask for these missing details before proceeding.\n"
        
        elif intent == 'show_order_summary':
            order_data = metadata.get('order_data', {})
            context += f"\n\n**CONTEXT:** All details collected. Show order summary:\n{self._format_order_summary(order_data)}\n\nAsk for final confirmation.\n"
        
        elif intent == 'detect_confirmation':
            order_data = metadata.get('order_data', {})
            context += f"""
**CRITICAL INTERNAL TASK - CONFIRMATION DETECTION**
You are now in confirmation detection mode. Analyze the customer's message to determine if they are confirming the order or not.

**Order Summary Being Confirmed:**
{self._format_order_summary(order_data)}

**Your Task:**
1. Carefully read the customer's message
2. Determine if they are confirming/agreeing (yes, ok, correct, sahi hai, theek hai, conform, proceed, etc.) OR if they want to change something/cancel
3. Set the internal status marker based on your analysis:
   - If customer is confirming: Include **INTERNAL_CONFIRMATION_STATUS: TRUE** in your response
   - If customer wants changes/is declining: Include **INTERNAL_CONFIRMATION_STATUS: FALSE** in your response

4. Then provide your normal customer-facing response.
"""
        return context
    
    def _format_order_summary(self, order_data: dict) -> str:
        """Format order data for display"""
        return f"""
Customer: {order_data.get('customer_name')}
Phone: {order_data.get('phone_number')}
Email: {order_data.get('email', 'N/A')}
Address: {order_data.get('address')}
Product: {order_data.get('product_name', order_data.get('product_url'))}
Quantity: {order_data.get('quantity', 1)}
"""

# ============================================================================
# WHATSAPP HANDLER
# ============================================================================

class WhatsAppHandler:
    """Handle WhatsApp messages via Evolution API"""
    
    def __init__(self, api_url: str, api_key: str, instance_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.instance_name = instance_name
        self.headers = {
            "apikey": api_key,
            "Content-Type": "application/json"
        }
    
    def send_message(self, phone_number: str, message: str, max_retries: int = 3) -> bool:
        """Send WhatsApp message with retry logic"""
        import requests
        from requests.exceptions import Timeout, ConnectionError
        
        phone_clean = phone_number.replace("+", "").replace("-", "").replace(" ", "")
        
        # Add country code if not present (assuming Indian numbers)
        if len(phone_clean) == 10:
            phone_clean = "91" + phone_clean
        
        url = f"{self.api_url}/message/sendText/{self.instance_name}"
        payload = {
            "number": phone_clean,
            "text": message
        }
        
        for attempt in range(max_retries):
            try:
                timeout = 15 + (attempt * 15)
                logger.info(f"📤 Sending message (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Message sent to {phone_number}")
                    return True
                else:
                    logger.warning(f"⚠️ Failed attempt {attempt + 1}: {response.status_code}")
                    if attempt < max_retries - 1: time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"⚠️ Error: {e}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt)
        return False
    
    def forward_media_with_base64(self, to_number: str, base64_data: str, caption: str, media_type: str = "image", max_retries: int = 3) -> bool:
        """Forward media using base64 data from Evolution API webhook"""
        import requests
        
        to_clean = to_number.replace("+", "").replace("-", "").replace(" ", "")
        if len(to_clean) == 10: to_clean = "91" + to_clean
        
        url = f"{self.api_url}/message/sendMedia/{self.instance_name}"
        payload = {
            "number": to_clean,
            "mediatype": media_type,
            "media": base64_data,
            "caption": caption
        }
        
        for attempt in range(max_retries):
            try:
                timeout = 20 + (attempt * 20)
                logger.info(f"📤 Forwarding {media_type} (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
                if response.status_code in [200, 201]:
                    logger.info(f"✅ {media_type.capitalize()} forwarded!")
                    return True
                else:
                    logger.warning(f"⚠️ Failed attempt {attempt + 1}: {response.status_code}")
                    if attempt < max_retries - 1: time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"⚠️ Error: {e}")
                if attempt < max_retries - 1: time.sleep(2 ** attempt)
        return False

    def send_media_via_url(self, to_number: str, media_url: str, caption: str, media_type: str = "image", max_retries: int = 3) -> bool:
        """Send media using direct URL"""
        import requests
        
        to_clean = to_number.replace("+", "").replace("-", "").replace(" ", "")
        if len(to_clean) == 10: to_clean = "91" + to_clean
        
        url = f"{self.api_url}/message/sendMedia/{self.instance_name}"
        payload = {
            "number": to_clean,
            "mediatype": media_type,
            "media": media_url,
            "caption": caption
        }
        
        for attempt in range(max_retries):
            try:
                timeout = 15 + (attempt * 10)
                logger.info(f"📤 Sending {media_type} via URL (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(url, json=payload, headers=self.headers, timeout=timeout)
                if response.status_code in [200, 201]:
                    logger.info(f"✅ {media_type.capitalize()} sent successfully!")
                    return True
                else:
                    logger.warning(f"⚠️ Failed attempt {attempt + 1}: {response.status_code}")
                    if attempt < max_retries - 1: time.sleep(1)
            except Exception as e:
                logger.warning(f"⚠️ Error: {e}")
                if attempt < max_retries - 1: time.sleep(1)
        return False

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)

# Initialize components
logger.info("🚀 Initializing WatchVine Multi-Agent Bot...")

conversation_manager = ConversationManager(MONGODB_URI, MONGODB_DB)
orchestrator = AgentOrchestrator(conversation_manager)
conversation_agent = ConversationAgent(
    api_key=GOOGLE_API_KEY,
    model=GOOGLE_MODEL,
    system_prompt=get_system_prompt(),
    conversation_manager=conversation_manager
)

# Initialize order storage
GOOGLE_APPS_SCRIPT_URL = os.getenv("GOOGLE_APPS_SCRIPT_URL")
GOOGLE_APPS_SCRIPT_SECRET = os.getenv("GOOGLE_APPS_SCRIPT_SECRET")

if GOOGLE_APPS_SCRIPT_URL and GOOGLE_APPS_SCRIPT_SECRET:
    logger.info("🔧 Initializing Google Apps Script for order storage...")
    order_storage = GoogleAppsScriptHandler(
        web_app_url=GOOGLE_APPS_SCRIPT_URL,
        secret_key=GOOGLE_APPS_SCRIPT_SECRET
    )
elif os.path.exists("credentials.json") and GOOGLE_SHEET_URL:
    logger.info("🔧 Initializing Google Sheets API for order storage...")
    order_storage = GoogleSheetsHandler(
        credentials_file="credentials.json",
        sheet_url=GOOGLE_SHEET_URL
    )
    order_storage.initialize_sheet_headers()
else:
    logger.warning("⚠️ Using MongoDB storage (Google Sheets not configured).")
    order_storage = MongoOrderStorage(mongodb_uri=MONGODB_URI, db_name=MONGODB_DB)

whatsapp = WhatsAppHandler(EVOLUTION_API_URL, EVOLUTION_API_KEY, INSTANCE_NAME)
conversation_agent.order_storage = order_storage

logger.info("✅ Multi-Agent Bot initialized successfully!")

# ============================================================================
# PRODUCT SEARCH HANDLER (AI-Based)
# ============================================================================

def detect_category_from_query(query: str) -> str:
    """Detect product category from search query"""
    query_lower = query.lower()
    category_map = {
        'mens_watch': ['men watch', 'mens watch', 'gent watch', 'boy watch'],
        'womens_watch': ['ladies watch', 'womens watch', 'women watch', 'girl watch', 'lady watch'],
        'mens_sunglasses': ['men sunglass', 'mens sunglass', 'men glass'],
        'womens_sunglasses': ['ladies sunglass', 'womens sunglass', 'women sunglass'],
        'wallet': ['wallet', 'purse'],
        'handbag': ['bag', 'handbag', 'hand bag'],
        'mens_shoes': ['men shoe', 'mens shoe', 'gent shoe'],
        'womens_shoes': ['ladies shoe', 'womens shoe', 'women shoe'],
        'loafers': ['loafer', 'formal shoe'],
        'flipflops': ['flipflop', 'flip flop', 'slipper'],
        'bracelet': ['bracelet', 'jewellery', 'jewelry']
    }
    
    for category_key, keywords in category_map.items():
        for keyword in keywords:
            if keyword in query_lower:
                return category_key
                
    if 'watch' in query_lower: return 'mens_watch'
    return None

def send_product_images_v2(keyword: str, phone_number: str, start_index: int = 0, batch_size: int = 10):
    """Search products and send limited batch"""
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if not keyword or len(keyword.strip()) < 2: return (False, 0, 0)
    
    logger.info(f"🔍 EXACT keyword search: '{keyword}' (batch: {start_index} to {start_index + batch_size})")
    
    try:
        category_key = detect_category_from_query(keyword)
        SEARCH_API_URL = os.getenv("TEXT_SEARCH_API_URL", "http://text_search_api:8001")
        search_payload = {"query": keyword, "max_results": 50}
        if category_key: search_payload["category_filter"] = category_key
        
        response = requests.post(f"{SEARCH_API_URL}/search/images-list", json=search_payload, timeout=30)
        
        if response.status_code != 200: return (False, 0, 0)
        
        result = response.json()
        if result.get('status') != 'success' or not result.get('products'): return (False, 0, 0)
        
        all_products = result.get('products', [])
        total_found = len(all_products)
        
        end_index = min(start_index + batch_size, total_found)
        products_to_send = all_products[start_index:end_index]
        
        # Send intro message
        intro_msg = f"Great! 🎉 Found {total_found} products for '{keyword}'\n"
        intro_msg += f"Showing {len(products_to_send)} products ({start_index+1}-{end_index})...\nPlease wait... 📸"
        whatsapp.send_message(phone_number, intro_msg)
        
        def send_single_product(idx, product):
            product_name = product.get('product_name', 'Unknown')
            price = product.get('price', 'N/A')
            product_url = product.get('product_url', '')
            images = product.get('images_base64', [])
            
            if not images: return False
            
            try:
                caption = f"📦 {product_name}\n💰 ₹{price}"
                if product_url: caption += f"\n🔗 {product_url}"
                if len(images) > 1: caption += f"\n\n📸 {len(images)} images available"
                
                return whatsapp.forward_media_with_base64(phone_number, images[0], caption, "image")
            except Exception: return False
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send_single_product, idx, prod) 
                      for idx, prod in enumerate(products_to_send, 1)]
            for future in as_completed(futures):
                if future.result(): success_count += 1
        
        # Cache products
        orchestrator.cache_product_data(phone_number, all_products)
        
        # Update sent_count
        try:
            orchestrator.conversation_manager.db.product_cache.update_one(
                {'phone_number': phone_number},
                {'$set': {'sent_count': end_index}}
            )
        except Exception as e: logger.error(f"Error updating sent_count: {e}")
        
        # Send completion message (UPDATED as per user request)
        if success_count > 0:
            completion_msg = ""
            
            if end_index < total_found:
                remaining = total_found - end_index
                completion_msg += f"📦 {remaining} more products available!\n"
                completion_msg += f"💬 Type 'show more {keyword}' to see them\n\n"
            
            # Append Gujarati Text (Transliterated instruction followed by Gujarati script)
            completion_msg += "તમે આ વૉચ બે રીતે ઓર્ડર કરી શકો છો\n"
            completion_msg += "1 અમદાવાદ-બોપલ સ્થિત અમારી સ્ટોર પરથી સીધી આવીને લઈ શકો છો.\n"
            completion_msg += "2. ઘર બેઠા Open Box Cash on Delivery દ્વારા પણ મંગાવી શકો છો.\n"
            completion_msg += "તમને કયો વિકલ્પ વધુ યોગ્ય લાગે છે? કૃપા કરીને જણાવશો."
            
            whatsapp.send_message(phone_number, completion_msg)
        
        return (True, total_found, success_count)
        
    except Exception as e:
        logger.error(f"❌ Error in product search: {e}")
        return (False, 0, 0)

# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages"""
    try:
        data = request.json
        if data.get('event') == 'messages.upsert':
            message_data = data.get('data', {})
            message_info = message_data.get('message', {})
            key_info = message_data.get('key', {})
            from_me = key_info.get('fromMe', False)
            remote_jid = key_info.get('remoteJid', '')
            
            if from_me or not remote_jid: return jsonify({"status": "success"}), 200
            
            phone_number = remote_jid.split('@')[0]
            
            # Handle Text Messages
            conversation = (
                message_info.get('conversation') or 
                message_info.get('extendedTextMessage', {}).get('text', '')
            )
            
            if not from_me and conversation:
                logger.info(f"📩 Message from {phone_number}: {conversation[:50]}...")
                
                # Analyze with Orchestrator (BackendToolClassifier)
                action, metadata = orchestrator.analyze_message(conversation, phone_number)
                logger.info(f"🎯 Action: {action}")
                
                if action == 'find_product':
                    keyword = metadata.get('keyword', '')
                    range_str = metadata.get('range', '0-10')
                    
                    try:
                        start_idx, end_idx = map(int, range_str.split('-'))
                        batch_size = end_idx - start_idx
                    except:
                        start_idx, batch_size = 0, 10
                    
                    # Check if already shown all
                    search_ctx = orchestrator.get_search_context(phone_number)
                    if search_ctx and search_ctx.get('sent_count', 0) >= search_ctx.get('total_found', 0) and search_ctx.get('total_found', 0) > 0:
                        all_shown_msg = f"✅ Tame badha {search_ctx['total_found']} products joi lidha che!\n\n"
                        all_shown_msg += f"🔍 Want to search for something else?\n"
                        whatsapp.send_message(phone_number, all_shown_msg)
                    else:
                        success, total_found, sent_count = send_product_images_v2(
                            keyword, phone_number, start_index=start_idx, batch_size=batch_size
                        )
                        if success:
                            orchestrator.save_search_context(phone_number, keyword, total_found, sent_count)
                        else:
                            whatsapp.send_message(phone_number, f"Sorry, no products found for '{keyword}' 😔")
                
                elif action == 'ai_chat' or action == 'ai_response':
                    response = conversation_agent.get_response(conversation, phone_number, metadata)
                    whatsapp.send_message(phone_number, response)
                
                elif action == 'save_order_direct' or action == 'save_data_to_google_sheet':
                    # Save order directly
                    order_data = metadata.get('order_data', metadata.get('data', {}))
                    order_data['order_id'] = f"WV{datetime.now().strftime('%Y%m%d%H%M%S')}{phone_number[-4:]}"
                    order_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    order_data['status'] = 'Pending'
                    
                    success = order_storage.save_order(order_data)
                    
                    if success:
                        response = f"🎉 *Order Confirmed!*
ID: {order_data['order_id']}\n"
                        response += "Our team will contact you within 24 hours. Thank you! 🙏"
                        whatsapp.send_message(phone_number, response)
                        orchestrator.clear_user_data(phone_number)
                    else:
                        whatsapp.send_message(phone_number, "Technical issue saving order. Please call us.")
                
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 STARTING WATCHVINE MAIN APPLICATION")
    app.run(host='0.0.0.0', port=5000, debug=False)