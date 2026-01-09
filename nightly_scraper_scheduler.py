"""
Nightly Scraper & Indexer Scheduler
Runs scraping and indexing automatically between 12 AM - 6 AM (India Time - IST)
Clears old data and rebuilds the index with fresh data daily
Works with any server timezone by converting to IST (UTC+5:30)
"""

import os
import sys
import time
import schedule
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
import subprocess
import pytz

# India Timezone (IST = UTC+5:30)
IST = pytz.timezone('Asia/Kolkata')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nightly_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "watchvine_refined")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "products")
SCRAPER_PATH = "fast_scraper.py"
INDEXER_PATH = "indexer.py"


class NightlyScraper:
    """Automated nightly scraper and indexer"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.is_running = False
    
    def clear_old_data(self):
        """Clear old product data from MongoDB"""
        try:
            logger.info("🗑️  Clearing old product data from MongoDB...")
            result = self.collection.delete_many({})
            logger.info(f"✅ Deleted {result.deleted_count} old products")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing old data: {str(e)}")
            return False
    
    def run_scraper(self):
        """Run the OPTIMIZED fast scraper to fetch fresh data (NO LIMITS - scrapes all products)"""
        try:
            logger.info("🕷️  Starting OPTIMIZED web scraper (5-8x FASTER, all products, no limits)...")
            logger.info("⚙️  Settings: 5 worker threads, 0.5-1s delays, auto cookie refresh")
            # Pass 'all' as limit, 'false' for clear_db (we already cleared it), and '5' for workers (optimized)
            result = subprocess.run(
                [sys.executable, SCRAPER_PATH, "all", "false", "5"],
                capture_output=True,
                text=True,
                timeout=None  # No timeout - let it complete for all 4000+ products
            )
            
            if result.returncode == 0:
                logger.info("✅ Scraper completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Scraper failed with code {result.returncode}")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running scraper: {str(e)}")
            return False
    
    def remove_old_index_files(self):
        """Remove old index files before creating new ones"""
        try:
            logger.info("🗑️  Removing old index files...")
            index_file = "image_identifier/vector_index.bin"
            metadata_file = "image_identifier/metadata.pkl"
            
            removed_count = 0
            if os.path.exists(index_file):
                os.remove(index_file)
                logger.info(f"  ✅ Removed: {index_file}")
                removed_count += 1
            
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"  ✅ Removed: {metadata_file}")
                removed_count += 1
            
            if removed_count == 0:
                logger.info("  ℹ️  No old index files found")
            else:
                logger.info(f"✅ Removed {removed_count} old index file(s)")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error removing old index files: {str(e)}")
            return False
    
    def run_indexer(self):
        """Run the indexer to rebuild FAISS index"""
        try:
            logger.info("🔨 Starting indexer...")
            result = subprocess.run(
                [sys.executable, INDEXER_PATH],
                capture_output=True,
                text=True,
                timeout=None  # No timeout - let it complete for all products
            )
            
            if result.returncode == 0:
                logger.info("✅ Indexer completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Indexer failed with code {result.returncode}")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running indexer: {str(e)}")
            return False
    
    def get_product_count(self):
        """Get current product count in database"""
        try:
            count = self.collection.count_documents({})
            return count
        except Exception as e:
            logger.error(f"❌ Error getting product count: {str(e)}")
            return 0
    
    def run_nightly_job(self):
        """Main nightly job - clear, scrape, and index"""
        if self.is_running:
            logger.warning("⚠️  Previous job still running, skipping...")
            return
        
        self.is_running = True
        start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info(f"🌙 NIGHTLY SCRAPER STARTED at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        # Step 1: Clear old data
        if not self.clear_old_data():
            logger.error("❌ Failed to clear old data, aborting job")
            self.is_running = False
            return
        
        # Step 2: Run scraper
        if not self.run_scraper():
            logger.error("❌ Scraper failed, aborting job")
            self.is_running = False
            return
        
        # Step 3: Verify products were scraped
        product_count = self.get_product_count()
        logger.info(f"📊 Total products in database: {product_count}")
        
        if product_count == 0:
            logger.error("❌ No products found after scraping, aborting indexing")
            self.is_running = False
            return
        
        # Step 4: Remove old index files
        if not self.remove_old_index_files():
            logger.warning("⚠️  Failed to remove old index files, continuing anyway...")
        
        # Step 5: Run indexer
        if not self.run_indexer():
            logger.error("❌ Indexer failed")
            self.is_running = False
            return
        
        # Job completed
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info(f"✅ NIGHTLY SCRAPER COMPLETED at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"📦 Total products indexed: {product_count}")
        logger.info("=" * 80)
        
        self.is_running = False
    
    def get_ist_time(self):
        """Get current time in IST (India Standard Time)"""
        return datetime.now(IST)
    
    def convert_ist_to_server_time(self, ist_hour, ist_minute=0):
        """Convert IST time to server's local time"""
        # Get current date in IST
        ist_now = self.get_ist_time()
        
        # Create IST datetime for the target time
        ist_target = ist_now.replace(hour=ist_hour, minute=ist_minute, second=0, microsecond=0)
        
        # Convert to server's local timezone
        server_local = ist_target.astimezone()
        
        return server_local.strftime("%H:%M")
    
    def start_scheduler(self):
        """Start the scheduler to run between 12 AM - 6 AM IST (India Time)"""
        
        # Convert IST midnight to server time
        server_time_midnight = self.convert_ist_to_server_time(0, 0)  # 12:00 AM IST
        
        # Get current server timezone name
        server_tz = datetime.now().astimezone().tzname()
        
        logger.info("🚀 Nightly Scraper Scheduler Started")
        logger.info(f"🌏 Target Timezone: IST (India Standard Time - UTC+5:30)")
        logger.info(f"🖥️  Server Timezone: {server_tz}")
        logger.info(f"⏰ Will run daily at 12:00 AM IST = {server_time_midnight} {server_tz}")
        logger.info("⏱️  Expected completion: ~2-3 hours (all 3514 products)")
        logger.info("📝 Logs saved to: nightly_scraper.log")
        logger.info("🔐 Auto cookie refresh: Every 10 minutes (never expires)")
        logger.info("🌐 Dynamic web_token: Extracted automatically")
        
        # Schedule job to run at converted server time
        schedule.every().day.at(server_time_midnight).do(self.run_nightly_job)
        
        # For testing: uncomment to run immediately
        # logger.info("🧪 Running test job immediately...")
        # self.run_nightly_job()
        
        logger.info("✅ Scheduler is running. Press Ctrl+C to stop.")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
                # Safety check: Don't run outside 12 AM - 6 AM window (IST)
                current_hour_ist = self.get_ist_time().hour
                if self.is_running and (current_hour_ist < 0 or current_hour_ist >= 6):
                    logger.warning(f"⚠️  Job running outside scheduled window (IST time: {current_hour_ist}:00), will complete current job")
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Scheduler error: {str(e)}")
                time.sleep(60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         NIGHTLY SCRAPER & INDEXER SCHEDULER                  ║
║         Automated Daily Product Database Updates             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    scraper = NightlyScraper()
    scraper.start_scheduler()
