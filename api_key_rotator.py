"""
API Key Rotation Manager
Automatically rotates between multiple Groq API keys to avoid rate limits
"""

import os
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from groq import Groq

logger = logging.getLogger(__name__)


class APIKeyRotator:
    """Manages multiple API keys and rotates between them"""
    
    def __init__(self, api_keys: List[str]):
        """
        Initialize API key rotator
        
        Args:
            api_keys: List of Groq API keys
        """
        self.api_keys = [key.strip() for key in api_keys if key.strip()]
        self.current_index = 0
        self.key_status = {}  # Track rate limit status
        
        if not self.api_keys:
            raise ValueError("No valid API keys provided")
        
        # Initialize status for all keys
        for key in self.api_keys:
            self.key_status[key] = {
                'available': True,
                'rate_limit_reset': None,
                'last_used': None,
                'error_count': 0,
                'banned': False,  # Track if key is permanently banned
                'ban_reason': None  # Store ban reason
            }
        
        logger.info(f"✅ API Key Rotator initialized with {len(self.api_keys)} keys")
    
    def get_next_key(self) -> str:
        """
        Get current active API key (SAME key until rate limit)
        
        Strategy: Use same key for all requests until it hits rate limit,
        then switch to next available key.
        
        Returns:
            str: Current active API key
        """
        # Try to find an available key
        attempts = 0
        while attempts < len(self.api_keys):
            key = self.api_keys[self.current_index]
            status = self.key_status[key]
            
            # Skip banned keys permanently
            if status.get('banned', False):
                logger.debug(f"⛔ Skipping banned key {self.current_index + 1}")
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                attempts += 1
                continue
            
            # Check if key is available
            if status['available']:
                # Check if rate limit has expired
                if status['rate_limit_reset']:
                    if datetime.now() > status['rate_limit_reset']:
                        # Rate limit expired, reset status
                        status['available'] = True
                        status['rate_limit_reset'] = None
                        status['error_count'] = 0
                        logger.info(f"🔄 API Key {self.current_index + 1} rate limit reset, using again")
                    else:
                        # Still rate limited, try next key
                        logger.debug(f"⏳ Key {self.current_index + 1} still rate limited, trying next...")
                        self.current_index = (self.current_index + 1) % len(self.api_keys)
                        attempts += 1
                        continue
                
                # Key is available, use it
                status['last_used'] = datetime.now()
                
                # Log only on first use or after switching
                if not hasattr(self, '_last_logged_key') or self._last_logged_key != self.current_index:
                    logger.info(f"🔑 Using API Key {self.current_index + 1}/{len(self.api_keys)} (will use until rate limit)")
                    self._last_logged_key = self.current_index
                
                # IMPORTANT: Don't move to next key - keep using same key until rate limited!
                # Only switch when mark_rate_limited() is called
                
                return self.api_keys[self.current_index]
            
            # Key not available, try next
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            attempts += 1
        
        # All keys rate limited, use first one anyway and let error handler deal with it
        logger.warning("⚠️ All API keys rate limited, waiting for reset...")
        return self.api_keys[0]
    
    def mark_rate_limited(self, api_key: str, retry_after: int = 60):
        """
        Mark an API key as rate limited and switch to next available key
        
        Args:
            api_key: The API key that hit rate limit
            retry_after: Seconds to wait before retry (default 60)
        """
        if api_key in self.key_status:
            # Find the index of the rate-limited key
            key_index = self.api_keys.index(api_key)
            
            # Mark as rate limited
            self.key_status[api_key]['available'] = False
            self.key_status[api_key]['rate_limit_reset'] = datetime.now() + timedelta(seconds=retry_after)
            self.key_status[api_key]['error_count'] += 1
            
            logger.warning(f"⚠️ API Key {key_index + 1}/{len(self.api_keys)} hit rate limit!")
            logger.warning(f"   Switching to next available key...")
            logger.warning(f"   This key will retry after {retry_after} seconds")
            
            # Move to next key (this is where we actually switch!)
            self.current_index = (key_index + 1) % len(self.api_keys)
            
            # Find next available key
            available_count = self._get_available_count()
            if available_count > 0:
                # Find the next available key index
                attempts = 0
                while attempts < len(self.api_keys):
                    test_key = self.api_keys[self.current_index]
                    if self.key_status[test_key]['available'] and not self.key_status[test_key].get('banned', False):
                        logger.info(f"✅ Switched to API Key {self.current_index + 1}/{len(self.api_keys)}")
                        break
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    attempts += 1
            
            logger.info(f"📊 Status: {available_count} available, {len(self.api_keys) - available_count} rate limited/banned")
    
    def mark_error(self, api_key: str):
        """
        Mark an API key as having an error
        
        Args:
            api_key: The API key that had an error
        """
        if api_key in self.key_status:
            self.key_status[api_key]['error_count'] += 1
            
            # If too many errors, temporarily disable
            if self.key_status[api_key]['error_count'] >= 3:
                logger.warning(f"⚠️ API Key has too many errors, disabling for 5 minutes")
                self.mark_rate_limited(api_key, retry_after=300)
    
    def mark_banned(self, api_key: str, reason: str = "organization_restricted"):
        """
        Mark an API key as permanently banned
        
        Args:
            api_key: The API key that is banned
            reason: Reason for ban (e.g., 'organization_restricted', 'invalid_key')
        """
        if api_key in self.key_status:
            self.key_status[api_key]['banned'] = True
            self.key_status[api_key]['ban_reason'] = reason
            self.key_status[api_key]['available'] = False
            
            # Find the key index for logging
            key_index = self.api_keys.index(api_key) + 1
            
            logger.error(f"🚫 API Key {key_index}/{len(self.api_keys)} marked as BANNED")
            logger.error(f"   Reason: {reason}")
            
            available_count = self._get_available_count()
            banned_count = sum(1 for status in self.key_status.values() if status.get('banned', False))
            
            logger.warning(f"📊 Status: {available_count} available, {banned_count} banned, {len(self.api_keys)} total")
            
            if available_count == 0:
                logger.critical("🚨 ALL API KEYS ARE BANNED OR UNAVAILABLE!")
                logger.critical("   Please add new API keys or contact Groq support")
    
    def _get_available_count(self) -> int:
        """Get count of available API keys"""
        return sum(1 for status in self.key_status.values() if status['available'])
    
    def get_status_report(self) -> str:
        """Get status report of all API keys"""
        report = f"\n{'='*60}\n"
        report += "API Key Status Report\n"
        report += f"{'='*60}\n"
        
        for i, key in enumerate(self.api_keys, 1):
            status = self.key_status[key]
            masked_key = f"{key[:8]}...{key[-4:]}"
            
            report += f"\nKey {i}: {masked_key}\n"
            
            # Show banned status first
            if status.get('banned', False):
                report += f"  Status: 🚫 BANNED\n"
                report += f"  Reason: {status.get('ban_reason', 'Unknown')}\n"
            elif status['available']:
                report += f"  Status: ✅ Available\n"
            else:
                report += f"  Status: ❌ Rate Limited\n"
            
            if status['rate_limit_reset']:
                time_left = (status['rate_limit_reset'] - datetime.now()).seconds
                report += f"  Reset in: {time_left}s\n"
            
            if status['last_used']:
                report += f"  Last used: {status['last_used'].strftime('%H:%M:%S')}\n"
            
            report += f"  Errors: {status['error_count']}\n"
        
        available_count = self._get_available_count()
        banned_count = sum(1 for status in self.key_status.values() if status.get('banned', False))
        
        report += f"\n{'='*60}\n"
        report += f"Available: {available_count}/{len(self.api_keys)}\n"
        report += f"Banned: {banned_count}/{len(self.api_keys)}\n"
        report += f"{'='*60}\n"
        
        return report


def load_api_keys_from_env() -> List[str]:
    """
    Load API keys from environment variables
    
    Supports:
    - GROQ_API_KEY_1, GROQ_API_KEY_2, ..., GROQ_API_KEY_10
    - GROQ_API_KEYS (comma-separated list)
    
    Returns:
        List of API keys
    """
    api_keys = []
    
    # Try numbered keys (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
    for i in range(1, 11):
        key = os.getenv(f"GROQ_API_KEY_{i}")
        if key:
            api_keys.append(key.strip())
            logger.info(f"✅ Loaded GROQ_API_KEY_{i}")
    
    # Try comma-separated list
    if not api_keys:
        keys_str = os.getenv("GROQ_API_KEYS")
        if keys_str:
            api_keys = [key.strip() for key in keys_str.split(",") if key.strip()]
            logger.info(f"✅ Loaded {len(api_keys)} keys from GROQ_API_KEYS")
    
    # Fallback to single key
    if not api_keys:
        single_key = os.getenv("GROQ_API_KEY")
        if single_key:
            api_keys = [single_key.strip()]
            logger.info(f"✅ Loaded single GROQ_API_KEY")
    
    if not api_keys:
        raise ValueError("No Groq API keys found in environment variables")
    
    logger.info(f"🔑 Total API keys loaded: {len(api_keys)}")
    return api_keys


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy keys
    test_keys = [
        "gsk_test_key_1",
        "gsk_test_key_2",
        "gsk_test_key_3"
    ]
    
    rotator = APIKeyRotator(test_keys)
    
    print("\n🧪 Testing API Key Rotation")
    print("="*60)
    
    # Simulate requests
    for i in range(5):
        key = rotator.get_next_key()
        print(f"Request {i+1}: Using key ending with ...{key[-4:]}")
    
    # Simulate rate limit
    print("\n⚠️ Simulating rate limit on first key")
    rotator.mark_rate_limited(test_keys[0], retry_after=10)
    
    # Continue requests
    print("\n🔄 Continuing with remaining keys")
    for i in range(5, 8):
        key = rotator.get_next_key()
        print(f"Request {i+1}: Using key ending with ...{key[-4:]}")
    
    # Print status
    print(rotator.get_status_report())
