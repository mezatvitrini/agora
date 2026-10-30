"""Agora - kategori etiketleyici."""
from __future__ import annotations
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
import requests
from config import RAW_CSV, PROCESSED_CSV, CATEGORIES, TAG_CACHE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.tagger")

LLM_API_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 3
MAX_CATEGORIES = 3
MAX_ACTORS = 8
CACHE_SAVE_EVERY = 10
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "Sen bir Türkçe haber analistisin. Verilen haber başlığını analiz edip "
    "yalnızca geçerli JSON döndür, başka açıklama yazma.\n"
    f"Kategoriler şu listeden seçilmeli: {', '.join(CATEGORIES)}.\n"
    f"Her başlığa 1 ile {MAX_CATEGORIES} arasında kategori ata; hiçbiri uymuyorsa boş liste döndür.\n"
    f"Aktörler: başlıkta adı geçen kişi, kurum, ülke veya gruplar (en fazla {MAX_ACTORS}).\n"
    'JSON şeması: {"kategoriler": ["..."], "aktorler": ["..."]}'
)


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _api_key() -> str | None:
    _load_env(PROJECT_ROOT / ".env")
    return os.environ.get("DEEPSEEK_API_KEY")


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _clean_result(data: dict) -> dict:
    allowed = set(CATEGORIES)
    categories: list[str] = []
    for item in data.get("kategoriler") or []:
        name = str(item).strip().lower()
        if name in allowed and name not in categories:
            categories.append(name)
    actors: list[str] = []
    for item in data.get("aktorler") or []:
        name = str(item).strip()
        if name and name not in actors:
            actors.append(name)
    return {"kategoriler": categories[:MAX_CATEGORIES], "aktorler": actors[:MAX_ACTORS]}


def tag_with_llm(title: str) -> dict:
    # Hata durumunda "_error" anahtarı eklenir; çağıran taraf bu sonucu önbelleğe almamalıdır.
    empty = {"kategoriler": [], "aktorler": [], "_error": True}
    title = (title or "").strip()
    if not title:
        return empty
    key = _api_key()
    if not key:
        logger.error("DEEPSEEK_API_KEY bulunamadı (.env veya ortam değişkeni)")
        return empty
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": title},
        ],
        "temperature": 0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("LLM isteği başarısız (%d/%d): %s", attempt + 1, LLM_MAX_RETRIES, exc)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("LLM geçici hata %s (%d/%d)", resp.status_code, attempt + 1, LLM_MAX_RETRIES)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            logger.error("LLM hatası %s: %s", resp.status_code, resp.text[:200])
            return empty
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError):
            logger.warning("LLM yanıtı beklenen biçimde değil")
            return empty
        data = _extract_json(content)
        if data is None:
            logger.warning("LLM geçersiz JSON döndürdü: %s", content[:120])
            return empty
        return _clean_result(data)
    logger.warning("LLM isteği %d denemede tamamlanamadı: %s", LLM_MAX_RETRIES, title[:60])
    return empty


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Önbellek okunamadı, sıfırdan başlanıyor: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Önbellek biçimi beklenmedik, yok sayılıyor")
        return {}
    return data


def _save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(path)


def _cache_key(row: dict) -> str:
    return (row.get("dedup_hash") or "").strip() or f"t:{row.get('title', '').strip().lower()}"


def main() -> None:
    input_path = Path(RAW_CSV)
    if not input_path.exists():
        logger.error("Girdi bulunamadı: %s", input_path)
        return
    if not _api_key():
        logger.error("DEEPSEEK_API_KEY yok, etiketleme atlandı")
        return
    output_path = Path(PROCESSED_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(TAG_CACHE)
    cache = _load_cache(cache_path)
    with input_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    hits = 0
    calls = 0
    failures = 0
    unsaved = 0
    for index, row in enumerate(rows, start=1):
        key = _cache_key(row)
        if key in cache:
            result = cache[key]
            hits += 1
        else:
            result = tag_with_llm(row.get("title", ""))
            if result.get("_error"):
                failures += 1
            else:
                cache[key] = result
                unsaved += 1
                if unsaved >= CACHE_SAVE_EVERY:
                    _save_cache(cache_path, cache)
                    unsaved = 0
            calls += 1
        row["kategoriler"] = json.dumps(result.get("kategoriler", []), ensure_ascii=False)
        row["aktorler"] = json.dumps(result.get("aktorler", []), ensure_ascii=False)
        if index % 20 == 0:
            logger.info("Etiketleniyor: %d/%d", index, len(rows))
    _save_cache(cache_path, cache)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Önbellek: %d isabet, %d LLM çağrısı, %d hata", hits, calls, failures)
    logger.info("Etiketleme tamamlandı: %s (%d satır)", output_path, len(rows))


if __name__ == "__main__":
    main()
