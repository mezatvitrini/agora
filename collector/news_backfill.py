"""Agora - The News API ile gecmis veri toplama (gun-konu bazli, devam edilebilir).

Kurallar:
- Gunluk kotaya saygi: her yanitin x-usagelimit-remaining basligi okunur, esik altina inince durur.
- Dakikada 30 siniri icin istekler arasi en az 3 sn bekler.
- 429/403/limit hatasinda durur, tekrar denemez.
- Tamamlanan gun-konu ciftleri data/backfill/news_progress.json'a yazilir; tekrar calistirmada atlanir.
- published degeri istenen gune uymayan satirlar sayilir ama CSV'ye yazilmaz.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from config import FEEDS, NEWS_API_KEY, NEWS_API_URL
from collector.rss_fetcher import _normalize_title, _similar_titles

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.backfill")

# --- Ayarlar ---
# The News API geçmişi ~180 günle sınırlı: 2026-05-05 öncesi sorgular boş dönüyor
# (2026-02-01…2026-05-01 arası doğrulandı: found=0). Bu yüzden varsayılan başlangıç 180 gün öncesi.
BASLANGIC = date(2026, 5, 5)
BITIS = date.today()
KONULAR = {
    "bitcoin": "bitcoin",
    "oil": "oil price OR crude oil",
    "gold": "gold price",
}
HAFTA_ICI = {0, 1, 2, 3, 4}          # oil/gold yalnizca hafta ici
DIL = "en"
SORGU_LIMIT = 3                       # plan siniri: istek basina en fazla 3 haber
ISTEK_ARASI_SN = 3.0                  # dakikada 30 siniri icin
KOTA_ESIGI = 5                        # kalan istek bunun altina inerse dur
MAKS_ISTEK = 90                       # bu calistirmadaki butce
ZAMAN_ASIMI = 20
OZET_MAX = 300

PROGRESS_PATH = Path("data/backfill/news_progress.json")
CSV_PATH = Path("data/backfill/news_historical.csv")
CSV_COLUMNS = ["published", "title", "description", "source", "ulke", "dil", "url",
               "konu", "kaynak_tipi"]


class KotaBitti(RuntimeError):
    pass


class LimitHatasi(RuntimeError):
    pass


@dataclass
class Kayit:
    published: str
    title: str
    description: str
    source: str
    ulke: str
    dil: str
    url: str
    konu: str
    kaynak_tipi: str


def gun_konu_ciftleri(baslangic: date = BASLANGIC) -> list[tuple[str, date]]:
    """Kronolojik sirada (konu, gun) ciftleri: bitcoin her gun, oil/gold hafta ici."""
    ciftler: list[tuple[str, date]] = []
    gun = baslangic
    while gun <= BITIS:
        for konu in KONULAR:
            if konu in ("oil", "gold") and gun.weekday() not in HAFTA_ICI:
                continue
            ciftler.append((konu, gun))
        gun += timedelta(days=1)
    return ciftler


def anahtar(konu: str, gun: date) -> str:
    return f"{gun.isoformat()}|{konu}"


def progress_yukle() -> dict:
    if not PROGRESS_PATH.exists():
        return {"tamamlanan": [], "istek": 0, "guncelleme": ""}
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Progress okunamadi, sifirdan baslanacak: %s", exc)
        return {"tamamlanan": [], "istek": 0, "guncelleme": ""}
    data.setdefault("tamamlanan", [])
    data.setdefault("istek", 0)
    data.setdefault("guncelleme", "")
    return data


def progress_yaz(data: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["guncelleme"] = datetime.now(timezone.utc).isoformat()
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, PROGRESS_PATH)


def csv_hazirla() -> tuple[set[str], dict[str, list[str]]]:
    """Mevcut CSV'yi okur: URL kumesi ve konu bazli normalize baslik indeksi dondurur."""
    if not CSV_PATH.exists():
        return set(), {}
    urls: set[str] = set()
    basliklar: dict[str, list[str]] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            url = (row.get("url") or "").strip()
            if url:
                urls.add(url)
            konu = (row.get("konu") or "").strip()
            basliklar.setdefault(konu, []).append(_normalize_title(row.get("title") or ""))
    return urls, basliklar


def satir_ekle(kayitlar: list[Kayit]) -> None:
    if not kayitlar:
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    yeni_dosya = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    for deneme in range(5):
        try:
            with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                if yeni_dosya:
                    writer.writeheader()
                for kayit in kayitlar:
                    writer.writerow(kayit.__dict__)
            return
        except PermissionError:
            if deneme == 4:
                raise
            time.sleep(0.3 * (deneme + 1))


def meta_ve_kota(resp: requests.Response) -> tuple[dict, int | None]:
    meta = {}
    try:
        meta = resp.json().get("meta") or {}
    except ValueError:
        meta = {}
    kalan = resp.headers.get("x-usagelimit-remaining")
    try:
        kalan_int = int(kalan) if kalan is not None else None
    except ValueError:
        kalan_int = None
    return meta, kalan_int


def istek_at(konu: str, gun: date) -> tuple[list[dict], int | None, dict]:
    """Tek gunluk pencere icin tek istek. Limit/limit-hatasi durumlarini yukari firlatir."""
    params = {
        "api_token": NEWS_API_KEY,
        "search": KONULAR[konu],
        # Tarih-only deger (YYYY-AA-GG) tek gunluk pencerede 0 sonuc donduruyor;
        # API tarih-saat bekliyor: D 00:00:00 -> D+1 00:00:00.
        "published_after": f"{gun.isoformat()}T00:00:00",
        "published_before": f"{(gun + timedelta(days=1)).isoformat()}T00:00:00",
        "language": DIL,
        "sort": "published_at",
        "limit": SORGU_LIMIT,
        "page": 1,
    }
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=ZAMAN_ASIMI,
                            headers={"Accept": "application/json"})
    except requests.RequestException as exc:
        raise RuntimeError(f"istek gonderilemedi: {exc}") from exc
    if resp.status_code in (429, 403):
        raise LimitHatasi(f"HTTP {resp.status_code} {resp.reason} | {resp.text[:500]}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} {resp.reason} | {resp.text[:500]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"JSON olmayan yanit | {resp.text[:300]}") from exc
    if payload.get("errors"):
        raise LimitHatasi(f"API hatasi: {payload['errors']}")
    meta, kalan = meta_ve_kota(resp)
    return payload.get("data") or [], kalan, meta


def kayda_cevir(ham: dict, konu: str) -> Kayit | None:
    title = (ham.get("title") or "").strip()
    url = (ham.get("url") or "").strip()
    published = (ham.get("published_at") or "").strip()
    if not title or not url or not published:
        return None
    source = (ham.get("source") or "").strip()
    feed = FEEDS.get(source) if isinstance(FEEDS.get(source), dict) else None
    return Kayit(
        published=published,
        title=title,
        description=(ham.get("description") or "").strip()[:OZET_MAX],
        source=source,
        ulke=(feed or {}).get("ulke", "") if feed else "",
        dil=(ham.get("language") or DIL).strip(),
        url=url,
        konu=konu,
        kaynak_tipi=(feed or {}).get("kaynak_tipi", "") if feed else "bilinmiyor",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agora - The News API gecmis veri toplama")
    parser.add_argument("--max-istek", type=int, default=MAKS_ISTEK,
                        help=f"bu calistirmada en fazla kac API istegi (varsayilan {MAKS_ISTEK})")
    parser.add_argument("--baslangic", default=BASLANGIC.isoformat(),
                        help=f"ilk gun (YYYY-AA-GG, varsayilan {BASLANGIC.isoformat()}); "
                             "The News API gecmisi ~180 gunle sinirli")
    args = parser.parse_args(argv)
    maks_istek = max(1, args.max_istek)
    try:
        baslangic = date.fromisoformat(args.baslangic)
    except ValueError:
        logger.error("Gecersiz --baslangic: %s", args.baslangic)
        return 2

    if not NEWS_API_KEY:
        logger.error("NEWS_API_KEY tanimli degil (.env). Toplama baslatilmadi.")
        return 2

    ciftler = gun_konu_ciftleri(baslangic)
    progress = progress_yukle()
    tamamlanan = set(progress["tamamlanan"])
    kalan_ciftler = [c for c in ciftler if anahtar(c[0], c[1]) not in tamamlanan]
    logger.info("Toplam gun-konu: %d | tamamlanmis: %d | kalan: %d",
                len(ciftler), len(tamamlanan), len(kalan_ciftler))

    urls, basliklar = csv_hazirla()
    istek = 0
    yeni_baslik = 0
    aralik_disi = 0
    konu_dagilim: dict[str, int] = {}
    bu_calismada: list[str] = []
    kalan_kota: int | None = None
    durus_nedeni = f"butce doldu ({maks_istek} istek)"

    for konu, gun in kalan_ciftler:
        if istek >= maks_istek:
            break
        if kalan_kota is not None and kalan_kota < KOTA_ESIGI:
            durus_nedeni = f"gunluk kota esigi (kalan {kalan_kota})"
            logger.warning("Kota esigi: kalan %s, duruldu. Yarin devam.", kalan_kota)
            break
        if istek:
            time.sleep(ISTEK_ARASI_SN)
        try:
            data, kalan_kota, meta = istek_at(konu, gun)
        except LimitHatasi as exc:
            durus_nedeni = f"limit hatasi: {exc}"
            logger.error("LIMIT HATASI, duruldu: %s", exc)
            break
        except RuntimeError as exc:
            durus_nedeni = f"istek hatasi: {exc}"
            logger.error("Istek hatasi, duruldu: %s", exc)
            break
        istek += 1

        yeni_kayitlar: list[Kayit] = []
        for ham in data:
            kayit = kayda_cevir(ham, konu)
            if kayit is None:
                continue
            if kayit.published[:10] != gun.isoformat():
                aralik_disi += 1
                continue
            if kayit.url in urls:
                continue
            normalized = _normalize_title(kayit.title)
            mevcut = basliklar.setdefault(konu, [])
            if any(_similar_titles(normalized, other) for other in mevcut):
                continue
            urls.add(kayit.url)
            mevcut.append(normalized)
            yeni_kayitlar.append(kayit)

        satir_ekle(yeni_kayitlar)
        yeni_baslik += len(yeni_kayitlar)
        konu_dagilim[konu] = konu_dagilim.get(konu, 0) + len(yeni_kayitlar)
        tamamlanan.add(anahtar(konu, gun))
        bu_calismada.append(anahtar(konu, gun))
        progress["tamamlanan"] = sorted(tamamlanan)
        progress["istek"] = progress.get("istek", 0) + 1
        progress_yaz(progress)
        logger.info("%s %s | ham %d, yeni %d, kalan kota %s",
                    gun.isoformat(), konu, len(data), len(yeni_kayitlar), kalan_kota)

    kalan_son = [c for c in ciftler if anahtar(c[0], c[1]) not in tamamlanan]
    print("\n=== RAPOR ===")
    print("kullanilan_istek:", istek)
    print("yeni_baslik:", yeni_baslik)
    print("konu_dagilim:", konu_dagilim)
    print("aralik_disi_satir:", aralik_disi)
    print("bu_calismada_tamamlanan:", len(bu_calismada))
    print("kalan_kota:", kalan_kota)
    print("kalan_gun_konu:", len(kalan_son))
    if yeni_baslik:
        gunde = istek and (len(bu_calismada) / istek) or 1.0
        print("gun_basina_cift:", round(gunde, 2))
        print("tahmini_kalan_calisma:", round(len(kalan_son) / gunde) if gunde else "-")
    print("durus_nedeni:", durus_nedeni)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
