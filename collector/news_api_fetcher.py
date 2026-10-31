"""Agora - The News API (thenewsapi.com) toplayıcı."""
from __future__ import annotations
import csv
import hashlib
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import requests
from config import NEWS_API_KEY, NEWS_API_URL, RAW_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.newsapi")

TIMEOUT_SECONDS = 20
DEFAULT_LIMIT = 3          # The News API ücretsiz planda istek başına üst sınır 3
DEFAULT_MAX_PAGES = 10
SUMMARY_MAX_CHARS = 300
CSV_COLUMNS = ["source", "title", "link", "published", "fetched_at", "dedup_hash",
               "ulke", "dil", "kaynak_tipi", "summary"]


class NewsAPIError(RuntimeError):
    """The News API'den dönen hata (HTTP durum kodu ve tam gövde ile)."""


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
    summary: str = ""


def _make_hash(source: str, title: str, link: str) -> str:
    raw = f"{source}|{title.strip().lower()}|{link.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _clean_summary(value: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())[:limit].strip()


def _request(params: dict) -> dict:
    """Tek bir API çağrısı; HTTP/JSON hatalarında tam gövdeyi NewsAPIError içinde döndürür."""
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=TIMEOUT_SECONDS,
                            headers={"Accept": "application/json"})
    except requests.RequestException as exc:
        raise NewsAPIError(f"İstek gönderilemedi: {exc}") from exc
    text = resp.text
    if resp.status_code != 200:
        raise NewsAPIError(f"HTTP {resp.status_code} — {resp.reason} | gövde: {text[:1000]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise NewsAPIError(f"Yanıt JSON değil | gövde: {text[:1000]}") from exc
    if isinstance(payload, dict) and payload.get("errors"):
        raise NewsAPIError(f"API hatası: {payload['errors']}")
    return payload


def fetch_news(query: str, from_date: str, to_date: str, language: str = "en",
               limit: int = DEFAULT_LIMIT, max_pages: int = DEFAULT_MAX_PAGES) -> list[Headline]:
    """The News API'den haber çeker; sayfalama yapar, sonuçları Headline listesi olarak döndürür.

    from_date / to_date: 'YYYY-MM-DD' (published_after / published_before olarak gönderilir).
    Hata durumunda NewsAPIError yükseltir (mesaj tam gövdeyi içerir).
    """
    if not NEWS_API_KEY:
        raise NewsAPIError("NEWS_API_KEY tanımlı değil (config.py / .env).")
    now = datetime.now(timezone.utc).isoformat()
    collected: list[Headline] = []
    seen_links: set[str] = set()
    found = None
    page = 1
    while page <= max_pages:
        payload = _request({
            "api_token": NEWS_API_KEY,
            "search": query,
            "published_after": from_date,
            "published_before": to_date,
            "language": language,
            "sort": "published_at",
            "limit": limit,
            "page": page,
        })
        meta = payload.get("meta") or {}
        found = meta.get("found", found)
        data = payload.get("data") or []
        logger.info("Sayfa %d: %d haber döndü (found=%s)", page, len(data), found)
        for item in data:
            title = (item.get("title") or "").strip()
            link = (item.get("url") or "").strip()
            if not title or not link or link in seen_links:
                continue
            seen_links.add(link)
            source = (item.get("source") or "thenewsapi").strip() or "thenewsapi"
            collected.append(Headline(
                source=source,
                title=title,
                link=link,
                published=item.get("published_at") or "",
                fetched_at=now,
                dedup_hash=_make_hash(source, title, link),
                ulke="",
                dil=(item.get("language") or language or "").strip(),
                kaynak_tipi="",
                summary=_clean_summary(item.get("description") or item.get("snippet") or ""),
            ))
        if len(data) < limit:
            break
        if isinstance(found, int) and len(collected) >= found:
            break
        page += 1
        time.sleep(0.2)
    logger.info("Toplam çekilen haber: %d (API found=%s)", len(collected), found)
    return collected


def _load_existing(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return ({r["dedup_hash"] for r in rows if r.get("dedup_hash")},
            {(r.get("link") or "").strip() for r in rows if r.get("link")})


def save_headlines(items: list[Headline], path: str = RAW_CSV) -> int:
    """Headline listesini RAW_CSV şemasıyla CSV'ye ekler; yeni satır sayısını döndürür."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_hashes, existing_links = _load_existing(output)
    new_items = [i for i in items
                 if i.dedup_hash not in existing_hashes and i.link not in existing_links]
    if not new_items:
        logger.info("Yeni satır yok (hepsi zaten CSV'de): %s", output)
        return 0
    write_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for item in new_items:
            writer.writerow(asdict(item))
    logger.info("CSV'ye yazıldı: %s (%d satır)", output, len(new_items))
    return len(new_items)


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "bitcoin"
    from_date = sys.argv[2] if len(sys.argv) > 2 else "2026-10-25"
    to_date = sys.argv[3] if len(sys.argv) > 3 else "2026-10-31"
    news = fetch_news(query, from_date=from_date, to_date=to_date, language="en")
    written = save_headlines(news)
    print(f"çekilen={len(news)} yazılan={written}")
