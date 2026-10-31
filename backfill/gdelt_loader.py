"""GDELT DOC 2.0 yükleyici.

- mode=ArtList, format=json, maxrecords=250, API anahtarı gerekmez.
- Yalnızca son ~3 ay; daha eski isteklerde uyarı verilir.
- Sorgular konu x dil x gün parçalarına bölünür; ilerleme data/backfill/progress.json'a yazılır,
  yarım kalan çalıştırma kaldığı yerden devam eder.
- Parçalar arası 5 sn beklenir; hata/rate limit durumunda üstel geri çekilme uygulanır, çökme yok.
- 250 kayda tam ulaşan parça "doymus" olarak işaretlenir.
- Sadece başlık, url, alan adı, ülke, dil, seendate ve konu saklanır; URL ve benzer başlıklar tekilleştirilir.
"""
from __future__ import annotations
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import requests

from backfill.topics import TOPICS
from config import FEEDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.gdelt")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKFILL_DIR = PROJECT_ROOT / "data" / "backfill"
PROGRESS_JSON = BACKFILL_DIR / "progress.json"
OUTPUT_CSV = BACKFILL_DIR / "headlines_historical.csv"

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
MAX_AGE_DAYS = 92
# GDELT: IP başına 5 sn kuralı; ceza penceresine girmemek için geniş pay bırakıyoruz.
SLEEP_BETWEEN = 10.0
RETRY_WAITS = (60, 120, 180, 240)
MAX_BACKOFF = 300
TIMEOUT = 40
CSV_COLUMNS = ["konu", "baslik", "url", "alan_adi", "kaynak", "kaynak_tipi",
               "ulke", "dil", "seendate"]

_TR_FOLD = str.maketrans({
    "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u", "ô": "o", "é": "e", "İ": "i",
})

# GDELT dil etiketleri -> ISO 639-1 kodu (config.FEEDS 'dil' alanıyla uyumlu)
DIL_ESLEME = {
    "turkish": "tr", "english": "en", "russian": "ru", "chinese": "zh",
    "arabic": "ar", "spanish": "es", "french": "fr", "german": "de",
    "portuguese": "pt", "italian": "it", "korean": "ko", "japanese": "ja",
}


def dil_kodu(dil: str) -> str:
    """'turkish' -> 'tr'; zaten kod ise olduğu gibi döner."""
    anahtar = str(dil or "").strip().lower()
    return DIL_ESLEME.get(anahtar, anahtar)


def _domain_source_map() -> dict[str, list[str]]:
    """config.FEEDS URL'lerinden alan adı -> kaynak(lar) eşlemesi."""
    esleme: dict[str, list[str]] = {}
    for source, meta in FEEDS.items():
        url = meta.get("url") if isinstance(meta, dict) else meta
        host = (urlparse(str(url)).hostname or "").lower()
        if not host:
            continue
        temiz = host[4:] if host.startswith("www.") else host
        esleme.setdefault(temiz, [])
        if source not in esleme[temiz]:
            esleme[temiz].append(source)
    return esleme


def _normalize_title(title: str) -> str:
    text = str(title or "").lower().replace("\u0307", "").translate(_TR_FOLD)
    return " ".join(re.sub(r"[^a-z0-9а-яёğüşıöç]+", " ", text).split())


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def load_progress() -> dict:
    if not PROGRESS_JSON.exists():
        return {"parcalar": {}, "doymus": []}
    try:
        return json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("İlerleme dosyası okunamadı, sıfırdan başlanıyor: %s", exc)
        return {"parcalar": {}, "doymus": []}


def save_progress(progress: dict) -> None:
    PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(PROGRESS_JSON)


def _chunks(days: int) -> list[tuple[date, date]]:
    """Son `days` günü günlük parçalara böler (en yeniden eskiye)."""
    bugun = date.today()
    parcalar = []
    for offset in range(days):
        gun = bugun - timedelta(days=offset)
        parcalar.append((gun, gun))
    return parcalar


def _istek(query: str, start: date, end: date) -> tuple[list[dict], str | None]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(MAX_RECORDS),
        "startdatetime": start.strftime("%Y%m%d") + "000000",
        "enddatetime": end.strftime("%Y%m%d") + "235959",
        "sort": "datedesc",
    }
    gecikme = RETRY_WAITS[0]
    for deneme in range(5):
        try:
            resp = requests.get(API_URL, params=params, timeout=TIMEOUT,
                                headers={"User-Agent": "AgoraBackfill/0.1"})
        except requests.RequestException as exc:
            logger.warning("GDELT isteği başarısız (%d/5): %s", deneme + 1, exc)
            time.sleep(gecikme)
            gecikme = min(gecikme * 2, MAX_BACKOFF)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("GDELT geçici hata %s (%d/5), %.0f sn bekleniyor",
                           resp.status_code, deneme + 1, gecikme)
            time.sleep(gecikme)
            gecikme = min(gecikme * 2, MAX_BACKOFF)
            continue
        if resp.status_code != 200:
            logger.error("GDELT hatası %s: %s", resp.status_code, resp.text[:160])
            return [], f"http_{resp.status_code}"
        try:
            veri = resp.json()
        except ValueError:
            # GDELT bazen boş gövde ya da düz metin döndürüyor
            if not resp.text.strip():
                return [], None
            logger.warning("GDELT yanıtı JSON değil: %s", resp.text[:120])
            return [], "json_hatasi"
        return veri.get("articles") or [], None
    return [], "deneme_tukendi"


def _kaynak_bul(domain: str, dil: str, esleme: dict[str, list[str]]) -> tuple[str, str | None]:
    """Alan adını config kaynağına eşler; belirsizse dili kullanır."""
    temiz = domain.lower()
    temiz = temiz[4:] if temiz.startswith("www.") else temiz
    adaylar = esleme.get(temiz, [])
    if not adaylar:
        return "bilinmiyor", None
    if len(adaylar) == 1:
        return adaylar[0], None
    # Aynı alan adı birden fazla feed (ör. BBC Türkçe / BBC World): dili eşle
    kod = dil_kodu(dil)
    for aday in adaylar:
        if FEEDS[aday].get("dil") == kod:
            return aday, None
    return adaylar[0], f"belirsiz_domain:{domain}"


def fetch_topic(konu: str, days: int = 3, diller: list[str] | None = None,
                sleep_between: float = SLEEP_BETWEEN) -> dict:
    """Bir konunun tüm dil sorgularını gün parçaları hâlinde çeker."""
    if days > MAX_AGE_DAYS:
        logger.warning("GDELT DOC 2.0 yalnızca son ~3 ayı destekler; %d gün istendi, "
                       "%d gün ile sınırlandırılıyor", days, MAX_AGE_DAYS)
        days = MAX_AGE_DAYS
    if konu not in TOPICS:
        raise SystemExit(f"Bilinmeyen konu: {konu} (geçerli: {', '.join(TOPICS)})")

    esleme = _domain_source_map()
    progress = load_progress()
    sonuc = {"konu": konu, "istek": 0, "hata": [], "doymus": [], "makale": 0, "benzersiz": 0}
    kayitlar: list[dict] = []

    sorgular = TOPICS[konu]["sorgular"]
    if diller:
        sorgular = {dil: sorgular[dil] for dil in diller if dil in sorgular}
        if not sorgular:
            raise SystemExit(f"{konu} için istenen diller yok (mevcut: "
                             f"{', '.join(TOPICS[konu]['sorgular'])})")

    for dil, query in sorgular.items():
        for bas, bit in _chunks(days):
            anahtar = f"{konu}|{dil}|{bas.isoformat()}"
            if anahtar in progress["parcalar"]:
                continue
            makaleler, hata = _istek(query, bas, bit)
            sonuc["istek"] += 1
            if hata:
                sonuc["hata"].append(f"{anahtar}: {hata}")
            if len(makaleler) >= MAX_RECORDS:
                sonuc["doymus"].append(anahtar)
                progress["doymus"].append(anahtar)
                logger.warning("Parça doymuş (250 kayıt): %s", anahtar)
            for makale in makaleler:
                domain = (makale.get("domain") or "").lower()
                dil_gdelt = dil_kodu(makale.get("language") or dil)
                kaynak, uyari = _kaynak_bul(domain, dil_gdelt, esleme)
                if uyari:
                    sonuc["hata"].append(uyari)
                kaynak_tipi = (FEEDS.get(kaynak, {}) or {}).get("kaynak_tipi", "bilinmiyor")
                kayitlar.append({
                    "konu": konu,
                    "baslik": (makale.get("title") or "").strip(),
                    "url": (makale.get("url") or "").strip(),
                    "alan_adi": domain,
                    "kaynak": kaynak,
                    "kaynak_tipi": kaynak_tipi or "bilinmiyor",
                    "ulke": (makale.get("sourcecountry") or "").strip(),
                    "dil": dil_gdelt,
                    "seendate": (makale.get("seendate") or "").strip(),
                })
            progress["parcalar"][anahtar] = {"adet": len(makaleler), "hata": hata,
                                             "zaman": datetime.now().isoformat(timespec="seconds")}
            if hata:
                # Hatalı parça "tamamlandı" sayılmaz: sonraki çalıştırmada yeniden denenir.
                progress["parcalar"].pop(anahtar, None)
            save_progress(progress)
            time.sleep(sleep_between)

    sonuc["makale"] = len(kayitlar)
    return {"ozet": sonuc, "kayitlar": kayitlar}


def dedupe(kayitlar: list[dict]) -> tuple[list[dict], int, int]:
    """URL ve benzer başlık tekrarını atar."""
    gorulen_url: set[str] = set()
    basliklar: list[str] = []
    benzersiz: list[dict] = []
    url_tekrar = baslik_tekrar = 0
    for kayit in kayitlar:
        url = kayit["url"].strip().lower()
        if url and url in gorulen_url:
            url_tekrar += 1
            continue
        normal = _normalize_title(kayit["baslik"])
        if normal and any(_similar(normal, mevcut) for mevcut in basliklar):
            baslik_tekrar += 1
            continue
        gorulen_url.add(url)
        basliklar.append(normal)
        benzersiz.append(kayit)
    return benzersiz, url_tekrar, baslik_tekrar


def append_csv(kayitlar: list[dict], path: Path | None = None) -> int:
    import csv

    hedef = Path(path) if path else OUTPUT_CSV
    hedef.parent.mkdir(parents=True, exist_ok=True)
    mevcut_url = set()
    var = hedef.exists()
    if var:
        with hedef.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mevcut_url.add((row.get("url") or "").strip().lower())
    yeni = [k for k in kayitlar if k["url"].strip().lower() not in mevcut_url]
    if not yeni:
        return 0
    with hedef.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not var:
            writer.writeheader()
        writer.writerows(yeni)
    return len(yeni)
