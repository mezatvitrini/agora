# Agora yapılandırma dosyası

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
