# Agora yapılandırma dosyası
from pathlib import Path
from dotenv import load_dotenv

# API anahtarları kodda tutulmaz; yalnızca .env dosyasından ortam değişkenine yüklenir.
# (load_dotenv mevcut ortam değişkenlerini ezmez.)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# RSS kaynakları (farklı sahiplik yapılarından)
FEEDS = {
    "aa": "https://www.aa.com.tr/tr/rss/default?cat=guncel",
    "bbc_turkce": "https://feeds.bbci.co.uk/turkce/rss.xml",
    "hurriyet": "https://www.hurriyet.com.tr/rss/anasayfa",
}

# Kategoriler
CATEGORIES = ["ekonomi", "guvenlik", "siyaset", "toplum", "spor"]

# Dosya yolları
RAW_CSV = "data/raw/headlines.csv"
PROCESSED_CSV = "data/processed/headlines_tagged.csv"
TAG_CACHE = "data/processed/tag_cache.json"
