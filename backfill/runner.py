"""Geri tarama çalıştırıcısı.

Kullanım:
  python -m backfill.runner --konu kripto --gun 3
  python -m backfill.runner --hepsi --gun 90
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from backfill.gdelt_loader import OUTPUT_CSV, append_csv, dedupe, fetch_topic
from backfill.topics import TOPICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.backfill")


def _mb(path: Path) -> float:
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agora - GDELT geri tarama")
    secim = parser.add_mutually_exclusive_group(required=True)
    secim.add_argument("--konu", choices=sorted(TOPICS), help="tek konu")
    secim.add_argument("--hepsi", action="store_true", help="tüm konular")
    parser.add_argument("--gun", type=int, default=3, help="kaç gün geriye (en fazla 92)")
    parser.add_argument("--dil", nargs="*", default=None, help="yalnızca bu diller (ör. en tr)")
    parser.add_argument("--bekleme", type=float, default=10.0, help="istekler arası bekleme (sn)")
    parser.add_argument("--cikti", default=str(OUTPUT_CSV))
    args = parser.parse_args(argv)

    konular = sorted(TOPICS) if args.hepsi else [args.konu]
    hedef = Path(args.cikti)

    toplam_istek = toplam_makale = toplam_benzersiz = 0
    url_tekrar = baslik_tekrar = 0
    hatalar: list[str] = []
    doymus: list[str] = []
    eklenen = 0

    for konu in konular:
        logger.info("Konu çekiliyor: %s (%d gün)", konu, args.gun)
        sonuc = fetch_topic(konu, days=args.gun, diller=args.dil, sleep_between=args.bekleme)
        ozet = sonuc["ozet"]
        kayitlar, url_t, baslik_t = dedupe(sonuc["kayitlar"])
        url_tekrar += url_t
        baslik_tekrar += baslik_t
        eklenen += append_csv(kayitlar, hedef)
        toplam_istek += ozet["istek"]
        toplam_makale += ozet["makale"]
        toplam_benzersiz += len(kayitlar)
        hatalar.extend(ozet["hata"])
        doymus.extend(ozet["doymus"])
        print(f"\n[{konu}] istek={ozet['istek']} makale={ozet['makale']} "
              f"benzersiz={len(kayitlar)} doymuş={len(ozet['doymus'])} hata={len(ozet['hata'])}")

    print("\n=== geri tarama özeti ===")
    print(f"  konu: {len(konular)} | istek: {toplam_istek} | çekilen makale: {toplam_makale}")
    print(f"  tekilleştirme: url tekrarı {url_tekrar}, benzer başlık {baslik_tekrar}")
    print(f"  benzersiz kayıt: {toplam_benzersiz} | CSV'ye eklenen yeni satır: {eklenen}")
    print(f"  doymuş parça (250 kayıt): {len(doymus)}" + (f" -> {', '.join(doymus[:5])}" if doymus else ""))
    print(f"  hata/uyarı: {len(hatalar)}" + (f" -> {hatalar[:5]}" if hatalar else ""))
    print(f"  dosya: {hedef} ({_mb(hedef):.2f} MB)")


if __name__ == "__main__":
    main()
