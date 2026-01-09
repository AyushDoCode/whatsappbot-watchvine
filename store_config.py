"""
Store Configuration
Complete WatchVine Store Information
"""

# Store contact information
STORE_NAME = "watchvine"
STORE_CONTACT_NUMBER = "9016220667"
STORE_ALTERNATE_NUMBER = ""
IMAGE_FORWARD_NUMBER = "9016220667"  # Number to forward images to
STORE_LOCATION = "Bopal haat complex, opp. sector 4, Sun City, Ahmedabad"
STORE_GOOGLE_MAP = "https://maps.app.goo.gl/miGV5wPVdXtdNgAN9"
STORE_TIMING = "Monday - Sunday: 2 PM - 8 PM"
STORE_VISIT_NOTE = "કૃપા કરીને visit પહેલાં 9016220667 પર ફોન કરીને આવજે (Kripa karke visit se pehle 9016220667 par phone karke aayein)"
STORE_INSTAGRAM = "https://www.instagram.com/watchvine01/"
STORE_WEBSITE = "https://watchvine01.cartpe.in/"

# Product Categories and Links
PRODUCT_CATEGORIES = {
    "Men's Watch": "https://watchvine01.cartpe.in/mens-watch.html",
    "Ladies Watch": "https://watchvine01.cartpe.in/ladies-watch-watches.html",
    "Sunglasses": "https://watchvine01.cartpe.in/sunglasses-eye-wear-men.html",
    "Hand Bags": "https://watchvine01.cartpe.in/hand-bags.html"
}

# Watch Brands and Links (Corrected URLs as per sheet)
WATCH_BRANDS = {
    "Fossil": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=fossi_l",
    "Tissot": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=tisso_t",
    "Armani": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=arman_i",
    "Tommy Hilfiger": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=tomm",
    "Rolex": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=role_x",
    "Rado": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=rad_o",
    "Omega": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=omeg_a",
    "Patek Philippe": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=Patek_Philippe",
    "Hublot": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=hublo",
    "Cartier": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=cartie",
    "Audemars Piguet (AP)": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=Audemars",
    "Tag Heuer": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=tag",
    "Michael Kors (MK)": "https://watchvine01.cartpe.in/allproduct.html?searchkeyword=mic"
}

# Products Available
AVAILABLE_PRODUCTS = [
    "Men's Watch", "Women's Watch", "Sunglasses", "Ladies Purse", 
    "Perfume", "Wallet", "Belt"
]

# Delivery Options
DELIVERY_OPTIONS = {
    "prepaid": {
        "name": "PREPAID DELIVERY",
        "charge": "₹100",
        "cod_charge": "No COD charges",
        "benefit": "Fast delivery"
    },
    "cod": {
        "name": "CASH ON DELIVERY",
        "advance": "₹200 advance pay for courier booking",
        "balance": "Balance amount at delivery time"
    }
}

# Address Format (Required Fields)
ADDRESS_FORMAT = {
    "name": "Name",
    "number": "Number",
    "landmark": "Landmark",
    "area": "Area / Society name",
    "pincode": "Pincode",
    "city": "City",
    "state": "State"
}

# Information bot CAN provide
AVAILABLE_INFO = [
    "Product selection",
    "Order collection",
    "Order confirmation",
    "Store location and timing",
    "Product categories and brands",
    "Delivery options"
]

# Information bot CANNOT provide (redirect to store contact)
REDIRECT_TO_STORE = [
    "Pricing details",
    "Gift packing options and charges",
    "Exact delivery timeline",
    "Return policy",
    "Warranty information",
    "Stock availability",
    "Product specifications (if not in conversation)",
    "Discount offers",
    "Bulk orders"
]

# Fallback response template
def get_fallback_response(query_topic: str = "") -> str:
    """Generate fallback response when information is not available"""
    if query_topic:
        return f"""I don't have detailed information about {query_topic}. For accurate information, please contact our store:

📞 Contact: {STORE_CONTACT_NUMBER}

Our team will be happy to assist you!"""
    else:
        return f"""For detailed information, please contact our store:

📞 Contact: {STORE_CONTACT_NUMBER}

Our team will provide you with all the details!"""
