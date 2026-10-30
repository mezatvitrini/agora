"""Agora - RSS toplayıcı."""
from __future__ import annotations
import csv
import hashlib
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import feedparser
import requests
from config import FEEDS, RAW_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.rss")

TIMEOUT_SECONDS = 15
USER_AGENT = "AgoraBot/0.1"
TITLE_SIMILARITY_THRESHOLD = 0.8
MIN_CONTAINMENT_TOKENS = 3
CSV_COLUMNS = ["source", "title", "link", "published", "fetched_at", "dedup_hash", "ulke", "dil", "kaynak_tipi"]
META_COLUMNS = ("ulke", "dil", "kaynak_tipi")
_TR_FOLD = str.maketrans({
    "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u", "ô": "o", "é": "e",
})

@dataclass
class Headline:
    source: str
    title: str
    link: str
    published: str
    fetched_at: str
    dedup_hash: str
    ulke: str = ""
    dil: str = ""
    kaynak_tipi: str = ""

def _feed_meta(source: str, value) -> dict:
    """FEEDS kaydını url/ulke/dil/kaynak_tipi sözlüğüne çevirir; eski metin biçimini de kabul eder."""
    if isinstance(value, dict):
        url = str(value.get("url") or "").strip()
        return {
            "url": url,
            "ulke": str(value.get("ulke") or "").strip(),
            "dil": str(value.get("dil") or "").strip(),
            "kaynak_tipi": str(value.get("kaynak_tipi") or "").strip(),
        }
    return {"url": str(value or "").strip(), "ulke": "", "dil": "", "kaynak_tipi": ""}

def _make_hash(source: str, title: str, link: str) -> str:
    raw = f"{source}|{title.strip().lower()}|{link.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def _fetch_feed(source: str, feed) -> list[Headline]:
    meta = _feed_meta(source, feed)
    url = meta["url"]
    if not url:
        logger.warning("Kaynak URL'si tanımsız: %s", source)
        return []
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Kaynak çekilemedi: %s (%s)", source, exc)
        return []
    parsed = feedparser.parse(resp.content)
    if parsed.bozo and not parsed.entries:
        logger.warning("Ayrıştırma hatası: %s", source)
        return []
    now = datetime.now(timezone.utc).isoformat()
    items = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue
        published = entry.get("published") or entry.get("updated") or ""
        items.append(Headline(
            source=source, title=title, link=link, published=published, fetched_at=now,
            dedup_hash=_make_hash(source, title, link),
            ulke=meta["ulke"], dil=meta["dil"], kaynak_tipi=meta["kaynak_tipi"],
        ))
    logger.info("Kaynak çekildi: %s (%d başlık)", source, len(items))
    return items

def _replace_with_retry(tmp: Path, target: Path) -> None:
    # Windows'ta virüs tarayıcı/indeksleyici dosyayı kısa süre kilitleyebiliyor.
    for attempt in range(5):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))

def _migrate_csv(path: Path) -> None:
    """Eski (dar) şemalı CSV'yi yeni sütunlarla yeniden yazar; satırlar korunur."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    if all(column in header for column in CSV_COLUMNS):
        return
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copyfile(path, backup)
        logger.info("Eski CSV yedeklendi: %s", backup)
    filled = 0
    for row in rows:
        meta = _feed_meta(row.get("source", ""), FEEDS.get(row.get("source", ""), {}))
        for column in META_COLUMNS:
            if not row.get(column):
                row[column] = meta.get(column, "")
                if row[column]:
                    filled += 1
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    _replace_with_retry(tmp, path)
    logger.info("CSV yeni şemaya taşındı: %d satır, %d alan config'ten dolduruldu", len(rows), filled)

def _load_existing_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["dedup_hash"] for row in csv.DictReader(fh)}

def _load_existing_links(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(row.get("link") or "").strip() for row in csv.DictReader(fh) if row.get("link")}

def _normalize_title(title: str) -> str:
    text = title.lower().replace("\u0307", "")
    text = text.translate(_TR_FOLD)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())

def _similar_titles(a: str, b: str) -> bool:
    if a == b:
        return True
    words_a, words_b = set(a.split()), set(b.split())
    if not words_a or not words_b:
        return False
    # Farklı sayılar (ör. "30 Ekim" / "29 Ekim") aynı başlığın farklı günlerde tekrarıdır, ayrı haber sayılır.
    # Kısaltılmış başlıkta sayı eksik olabileceği için yalnızca iki tarafta da sayı varsa ve çelişiyorsa engellenir.
    nums_a = {w for w in words_a if w.isdigit()}
    nums_b = {w for w in words_b if w.isdigit()}
    if nums_a and nums_b and not (nums_a <= nums_b or nums_b <= nums_a):
        return False
    if len(words_a & words_b) / len(words_a | words_b) >= TITLE_SIMILARITY_THRESHOLD:
        return True
    # Kapsama kuralı: kısa başlık, uzun başlığın kelimelerinin tamamını içeriyorsa aynı haber sayılır.
    shorter, longer = (words_a, words_b) if len(words_a) <= len(words_b) else (words_b, words_a)
    return len(shorter) >= MIN_CONTAINMENT_TOKENS and shorter <= longer

def _load_existing_titles(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    index: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            index.setdefault(row.get("source", ""), []).append(_normalize_title(row.get("title", "")))
    return index

def collect() -> None:
    output = Path(RAW_CSV)
    output.parent.mkdir(parents=True, exist_ok=True)
    _migrate_csv(output)
    existing = _load_existing_hashes(output)
    existing_titles = _load_existing_titles(output)
    existing_links = _load_existing_links(output)
    all_items = []
    empty_sources = []
    for source, value in FEEDS.items():
        items = _fetch_feed(source, value)
        if not items:
            empty_sources.append(source)
        all_items.extend(items)
    # Aynı başlık farklı kaynakta ayrı satır olarak kalır: Agora'nın amacı kaynakları karşılaştırmak.
    seen = set(existing)
    seen_links = set(existing_links)
    new_items = []
    duplicate_skipped = 0
    similar_skipped = 0
    link_skipped = 0
    for item in all_items:
        if item.dedup_hash in seen:
            duplicate_skipped += 1
            continue
        if item.link in seen_links:
            link_skipped += 1
            continue
        normalized = _normalize_title(item.title)
        candidates = existing_titles.get(item.source, [])
        if any(_similar_titles(normalized, other) for other in candidates):
            similar_skipped += 1
            logger.info("Benzer başlık atlandı: %s (%s)", item.title[:70], item.source)
            continue
        seen.add(item.dedup_hash)
        seen_links.add(item.link)
        new_items.append(item)
        existing_titles.setdefault(item.source, []).append(normalized)
    logger.info("Toplam: %d, yeni: %d, yinelenen: %d, benzer: %d, aynı URL: %d",
                len(all_items), len(new_items), duplicate_skipped, similar_skipped, link_skipped)
    if empty_sources:
        logger.warning("Veri gelmeyen kaynaklar: %s", ", ".join(empty_sources))
    if not new_items:
        return
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for item in new_items:
            writer.writerow(asdict(item))
    logger.info("CSV'ye yazıldı: %s (%d yeni satır)", output, len(new_items))

if __name__ == "__main__":
    collect()
