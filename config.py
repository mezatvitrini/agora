# Agora yapılandırma dosyası
from pathlib import Path
from dotenv import load_dotenv

# API anahtarları kodda tutulmaz; yalnızca .env dosyasından ortam değişkenine yüklenir.
# (load_dotenv mevcut ortam değişkenlerini ezmez.)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# RSS kaynakları
# Her kaynak: url, ulke (ISO 3166-1 alpha-2), dil (ISO 639-1),
# kaynak_tipi ("devlet_baglantili" / "ozel" / "ajans").
# Eski biçim (yalnızca URL metni) de desteklenir: collector eksik alanları boş bırakır.
FEEDS = {
    # --- Türkiye ---
    "aa": {
        "url": "https://www.aa.com.tr/tr/rss/default?cat=guncel",
        "ulke": "TR",
        "dil": "tr",
        "kaynak_tipi": "devlet_baglantili",
    },
    "bbc_turkce": {
        "url": "https://feeds.bbci.co.uk/turkce/rss.xml",
        "ulke": "GB",
        "dil": "tr",
        "kaynak_tipi": "devlet_baglantili",
    },
    "hurriyet": {
        "url": "https://www.hurriyet.com.tr/rss/anasayfa",
        "ulke": "TR",
        "dil": "tr",
        "kaynak_tipi": "ozel",
    },
    "dailysabah": {
        "url": "https://www.dailysabah.com/rssFeed/all",
        "ulke": "TR",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    # --- ABD ---
    "cnn_world": {
        "url": "http://rss.cnn.com/rss/edition_world.rss",
        "ulke": "US",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
    "nyt_world": {
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "ulke": "US",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
    # --- İngiltere / Avrupa ---
    "bbc_world": {
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "ulke": "GB",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    "euronews": {
        "url": "https://www.euronews.com/rss?level=theme&name=news",
        "ulke": "FR",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
    "guardian_world": {
        "url": "https://www.theguardian.com/world/rss",
        "ulke": "GB",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
    # --- Rusya ---
    "rt": {
        "url": "https://www.rt.com/rss/",
        "ulke": "RU",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    "tass": {
        "url": "https://tass.com/rss/v2.xml",
        "ulke": "RU",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    # --- Orta Doğu ---
    "aljazeera": {
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "ulke": "QA",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    # --- Çin ---
    "cgtn": {
        "url": "https://www.cgtn.com/subscribe/rss/section/world.xml",
        "ulke": "CN",
        "dil": "en",
        "kaynak_tipi": "devlet_baglantili",
    },
    # --- Kripto / piyasa ---
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "ulke": "US",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "ulke": "US",
        "dil": "en",
        "kaynak_tipi": "ozel",
    },
}

# Kategoriler
CATEGORIES = ["ekonomi", "guvenlik", "siyaset", "toplum", "spor"]

# --- Embedding tabanlı olay kümeleme (yerel, LLM çağrısı yok) ---
CLUSTER_THRESHOLD = 0.6
EXCLUDED_CATEGORIES = ["spor"]
EMBEDDING_EXCLUDE_SOURCES = ["coindesk", "cointelegraph"]  # ülke karşılaştırmasından hariç
CLUSTER_MAX_DAYS = 3
CLUSTER_MAX_SIZE = 15
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_DIR = "data/processed/embeddings"
CLUSTERS_CSV = "data/processed/clusters_embedding.csv"
GAPS_CSV = "data/processed/gaps_embedding.csv"

# Frame comparator entegrasyonu için hazır bayrak (henüz bağlanmadı, varsayılan kapalı).
# Açıldığında tek ülkeli kümeler LLM karşılaştırmasına gönderilmemelidir.
FRAME_SKIP_SINGLE_COUNTRY = False

# Dosya yolları
RAW_CSV = "data/raw/headlines.csv"
PROCESSED_CSV = "data/processed/headlines_tagged.csv"
TAG_CACHE = "data/processed/tag_cache.json"
