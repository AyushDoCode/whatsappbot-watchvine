"""
Backend Tool Classifier AI
Analyzes conversation and decides which tool to use
Returns JSON response: {tool: "ai_chat"} or {tool: "save_data_to_google_sheet", data: {...}}
Uses Google Gemini API with Context Caching
"""

import os
import json
import logging
import time
from datetime import datetime
import google.generativeai as genai
from google.generativeai import caching

logger = logging.getLogger(__name__)

class BackendToolClassifier:
    """
    Backend AI that classifies user intent and decides which tool to call
    This AI does NOT respond to user - it only decides actions
    Uses Google Gemini API
    """
    
    def __init__(self):
        """
        Initialize Backend Tool Classifier with Gemini
        """
        self.api_key = os.getenv("Google_api")
        if not self.api_key:
            logger.warning("⚠️ Google_api not found in environment variables. Please set it.")
            
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
        # Get model from env or use default
        env_model = os.getenv("google_model", "gemini-1.5-flash-001")
        # Ensure model name has 'models/' prefix if not present (Gemini API often prefers it)
        if not env_model.startswith("models/") and not env_model.startswith("gemini-"):
             self.model_name = f"models/{env_model}"
        else:
             self.model_name = env_model
             
        self.cache_name = "watchvine_classifier_cache_v1"
        self.cached_content = None
        self.last_cache_update = 0
        self.CACHE_TTL = 3600 # 1 hour refresh (cache lives longer, but we refresh ref locally)

        # Rate limit tracking
        self.last_request_time = {}
        self.min_request_interval = 1.0 
        
        logger.info(f"✅ Backend Classifier initialized with Gemini ({self.model_name})")

    def _get_static_instructions(self) -> str:
        """Returns the static part of the system prompt to be cached"""
        return """
WatchVine bot tool classifier.
You are an AI system that analyzes conversation history and decides the next action.
You do NOT generate chat responses. You ONLY return a JSON object indicating which tool to use.

TOOLS & OUTPUT RULES:

1. ai_chat
   JSON: {"tool": "ai_chat"}
   Use when:
   - User is greeting (Hi, Hello)
   - User asks general questions ("shop open?", "delivery time?")
   - User asks for categories without specific brand ("show watches", "bags dikhao")
   - User is just chatting
   - Search result pagination is complete ("All products shown")

2. find_product
   JSON: {"tool": "find_product", "keyword": "brand+type", "range": "X-Y"}
   Use when:
   - User asks for specific brand or product ("Rolex watch", "Gucci bag")
   - User asks to see "more" products after a search
   
   RANGE & PAGINATION RULES (CRITICAL):
   - "range" format is "Start-End" (0-based index)
   - First search: "0-10"
   - "Show more"/"Next"/"Aur dikhao": Increment range by 10 (e.g., 10-20, 20-30)
   - Check the provided "SEARCH INFO" in context to determine the next range.
   - Max range: Up to 300.
   - If User says "show more" but context shows all products sent: Return {"tool": "ai_chat"}
   
   KEYWORD RULES:
   - ALWAYS include the category type.
   - "Rolex" -> "rolex watch"
   - "Gucci" -> "gucci bag" (or context appropriate)
   - If user says "koi bhi" (any): Pick a popular brand + category (e.g., "gucci bag")
   - Keep the SAME keyword as previous search when paginating.

3. send_all_images
   JSON: {"tool": "send_all_images", "product_name": "exact name"}
   Use when:
   - User specifically asks for "all photos" or "baki images" of a SPECIFIC single product.
   - Example: "Rolex GMT ke sare photo bhejo" -> {"tool": "send_all_images", "product_name": "Rolex GMT"}

4. save_data_to_google_sheet
   JSON: {"tool": "save_data_to_google_sheet", "data": {...}}
   Use when:
   - Customer explicitly confirms order ("Confirm", "Book it", "Order this")
   - AND you have ALL details: Name, Phone, Address.
   - If details missing, return {"tool": "ai_chat"} to ask for them.

EXAMPLES:
Input: "rolex watch"
Output: {"tool": "find_product", "keyword": "rolex watch", "range": "0-10"}

Input: "show more" (Context: Last search 'rolex watch', sent 10/50)
Output: {"tool": "find_product", "keyword": "rolex watch", "range": "10-20"}

Input: "aur dikhao" (Context: Last search 'gucci bag', sent 140/150)
Output: {"tool": "find_product", "keyword": "gucci bag", "range": "140-150"}

Input: "more" (Context: Last search 'rolex watch', sent 150/150)
Output: {"tool": "ai_chat"}

Input: "watches chahiye"
Output: {"tool": "ai_chat"}

Input: "Rolex GMT ni badhi images"
Output: {"tool": "send_all_images", "product_name": "Rolex GMT"}

Return ONLY JSON.
"""

    def _get_or_create_cache(self):
        """Creates or retrieves cached content for system instructions"""
        if not self.api_key:
            return None
            
        current_time = time.time()
        
        # specific name for the cache
        cache_name = self.cache_name
        
        # If we have a valid local reference, return it
        if self.cached_content and (current_time - self.last_cache_update < self.CACHE_TTL):
            return self.cached_content

        try:
            # Check if cache exists (by iterating or specific name if API supported retrieval by name easily)
            # For simplicity in this implementation, we'll try to create it. 
            # If it exists, we might get an error or a new one. 
            # Ideally we list and find.
            
            # Listing caches to find ours
            existing_cache = None
            for c in caching.CachedContent.list():
                if c.display_name == cache_name:
                    existing_cache = c
                    break
            
            if existing_cache:
                # Update expiration? Or just use it.
                # existing_cache.update(ttl=datetime.timedelta(hours=2))
                logger.info(f"♻️ Using existing cache: {existing_cache.name}")
                self.cached_content = existing_cache
            else:
                # Create new cache
                logger.info("🆕 Creating new context cache...")
                system_instruction = self._get_static_instructions()
                
                self.cached_content = caching.CachedContent.create(
                    model=self.model_name,
                    display_name=cache_name,
                    system_instruction=system_instruction,
                    ttl=datetime.timedelta(hours=2) # Cache for 2 hours
                )
                logger.info(f"✅ Cache created: {self.cached_content.name}")
            
            self.last_cache_update = current_time
            return self.cached_content
            
        except Exception as e:
            logger.error(f"❌ Cache operation failed: {e}")
            return None

    def analyze_and_classify(self, conversation_history: list, user_message: str, phone_number: str, search_context: dict = None) -> dict:
        """
        Analyze conversation and return tool decision in JSON format
        """
        if not self.api_key:
            logger.error("❌ No API Key")
            return {"tool": "ai_chat"}

        # Rate limiting
        if phone_number in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[phone_number]
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time[phone_number] = time.time()

        # Build dynamic context
        context_str = self._build_context_string(conversation_history, user_message, search_context)
        
        try:
            # Try to use cache
            cached_content = self._get_or_create_cache()
            
            if cached_content:
                # Use model with cache
                model = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
                response = model.generate_content(
                    context_str,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
            else:
                # Fallback to non-cached standard request
                logger.warning("⚠️ Cache unavailable, using standard request")
                model = genai.GenerativeModel(self.model_name)
                full_prompt = self._get_static_instructions() + "\n\n" + context_str
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
            
            # Parse result
            result_text = response.text.strip()
            # Clean up markdown code blocks if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            logger.info(f"🔍 Classifier Decision: {result_text}")
            return json.loads(result_text)

        except Exception as e:
            logger.error(f"❌ Classifier Error: {e}")
            return {"tool": "ai_chat"}

    def _build_context_string(self, history: list, current_message: str, search_context: dict) -> str:
        """Builds the dynamic string for the request"""
        # Format history
        hist_str = ""
        for msg in history[-10:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            hist_str += f"{role.upper()}: {content}\n"
            
        # Format search info
        search_info = ""
        if search_context:
            keyword = search_context.get('keyword', '')
            sent_count = search_context.get('sent_count', 0)
            total_found = search_context.get('total_found', 0)
            
            if keyword and total_found > 0:
                next_start = sent_count
                next_end = sent_count + 10
                search_info = f"\n[SEARCH INFO]\nLast Keyword: '{keyword}'\nProducts Sent: {sent_count}/{total_found}\nNext Range Suggestion: '{next_start}-{next_end}'\n"

        return f"""
CONVERSATION HISTORY:
{hist_str}

{search_info}

CURRENT MESSAGE:
{current_message}
"""

    def extract_order_data_from_history(self, conversation_history: list, phone_number: str) -> dict:
        """
        Extract order data from conversation history
        Simple regex extraction as fallback or helper
        """
        order_data = {
            "customer_name": "",
            "phone_number": phone_number,
            "email": "",
            "address": "",
            "product_name": "",
            "product_url": "",
            "quantity": 1
        }
        
        # Simple extraction logic (similar to previous)
        for msg in conversation_history:
            content = msg.get('content', '').lower()
            if 'http' in content:
                 import re
                 url = re.search(r'https?://[^\s]+', content)
                 if url: order_data['product_url'] = url.group()
                 
        return order_data