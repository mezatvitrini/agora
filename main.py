"""Agora - ana çalıştırma dosyası."""
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collector.rss_fetcher import collect
from analyzer.category_tagger import main as tag
from analyzer.frame_comparator import main as compare_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agora - haber toplama ve çerçeveleme hattı")
    parser.add_argument("--cluster-mode", choices=["legacy", "embedding"], default="legacy",
                        help="Çerçeve karşılaştırması için kümeleme yöntemi (varsayılan: legacy)")
    args = parser.parse_args()

    print("1. RSS toplanıyor...")
    collect()
    print("2. Kategoriler etiketleniyor...")
    tag()
    print(f"  3. Çerçeve karşılaştırması yapılıyor... (mod: {args.cluster_mode})")
    compare_frames(["--cluster-mode", args.cluster_mode])
    if args.cluster_mode == "embedding":
        print("4. Alıntı doğrulama (yerel, ücretsiz)...")
        from analyzer.verify_quotes import main as verify_quotes
        verify_quotes([])
    print("[OK] Agora tamamlandı.")
