"""Agora - ana çalıştırma dosyası."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collector.rss_fetcher import collect
from analyzer.category_tagger import main as tag

if __name__ == "__main__":
    print("1. RSS toplanıyor...")
    collect()
    print("2. Kategoriler etiketleniyor...")
    tag()
    print("[OK] Agora tamamlandı.")
