
import os
from dotenv import load_dotenv
load_dotenv()

# Google Play Store App IDs
APP_IDS = {
# env
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    'BoAMobile': os.getenv('BoAMobile_APP_ID', 'com.boa.boaMobileBanking'),
    'DashenBank': os.getenv('DASHENBANK_APP_ID', 'com.dashen.dashensuperapp')

}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BoAMobile': 'BoA Mobile',
    'DashenBank': 'Dashen Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 400)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',
    'country': 'et'  # Ethiopia
}

# File Paths
DATA_PATHS = {
    'processed': '../data/processed', 
}


