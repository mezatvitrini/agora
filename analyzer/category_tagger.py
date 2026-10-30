"""Agora - kategori etiketleyici."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
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
# Varsayılan tek tek etiketleme; toplu gönderim için --batch-size N (N>1) kullanılır.
BATCH_SIZE = 1
BATCH_MAX_TOKENS = 900
SINGLE_MAX_TOKENS = 200
MAX_TITLE_CHARS = 300
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# deepseek-chat fiyatları (USD / 1M token; önbellek isabetsiz giriş ve çıkış)
COST_INPUT_PER_MTOK = 0.27
COST_OUTPUT_PER_MTOK = 1.10

SYSTEM_PROMPT = (
    "Sen bir Türkçe haber analistisin. Verilen haber başlığını analiz edip "
    "yalnızca geçerli JSON döndür, başka açıklama yazma.\n"
    f"Kategoriler şu listeden seçilmeli: {', '.join(CATEGORIES)}.\n"
    f"Her başlığa 1 ile {MAX_CATEGORIES} arasında kategori ata; hiçbiri uymuyorsa boş liste döndür.\n"
    f"Aktörler: başlıkta adı geçen kişi, kurum, ülke veya gruplar (en fazla {MAX_ACTORS}).\n"
    'JSON şeması: {"kategoriler": ["..."], "aktorler": ["..."]}'
)

BATCH_SYSTEM_PROMPT = (
    "Sen bir Türkçe haber analistisin. Sana numaralı haber başlıkları verilir. "
    "Her başlığı ayrı ayrı analiz et ve yalnızca geçerli JSON döndür, başka açıklama yazma.\n"
    f"Kategoriler şu listeden seçilmeli: {', '.join(CATEGORIES)}.\n"
    f"Her başlığa 1 ile {MAX_CATEGORIES} arasında kategori ata; hiçbiri uymuyorsa boş liste döndür.\n"
    f"Aktörler: başlıkta adı geçen kişi, kurum, ülke veya gruplar (en fazla {MAX_ACTORS}).\n"
    'JSON şeması: {"sonuclar": [{"id": 1, "kategoriler": ["..."], "aktorler": ["..."]}]}\n'
    "Verilen her numara için tam bir kayıt döndür; 'id' alanı başlığın numarası olmalı."
)

# İstek/token sayaçları (main() başında sıfırlanır)
TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usage_missing": 0}
# Hata/deneme sayaçları (main() başında sıfırlanır)
RUN_STATS = {"retries": 0, "failed_requests": 0, "bad_json": 0}


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


def _post(messages: list[dict], max_tokens: int) -> tuple[str | None, dict]:
    """LLM'e tek istek atar; (içerik, usage) döndürür. Hata halinde (None, {})."""
    key = _api_key()
    if not key:
        logger.error("DEEPSEEK_API_KEY bulunamadı (.env veya ortam değişkeni)")
        return None, {}
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT)
        except requests.RequestException as exc:
            RUN_STATS["retries"] += 1
            logger.warning("LLM isteği başarısız (%d/%d): %s", attempt + 1, LLM_MAX_RETRIES, exc)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            RUN_STATS["retries"] += 1
            logger.warning("LLM geçici hata %s (%d/%d)", resp.status_code, attempt + 1, LLM_MAX_RETRIES)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            RUN_STATS["failed_requests"] += 1
            logger.error("LLM hatası %s: %s", resp.status_code, resp.text[:200])
            return None, {}
        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError):
            RUN_STATS["failed_requests"] += 1
            logger.warning("LLM yanıtı beklenen biçimde değil")
            return None, {}
        usage = body.get("usage") or {}
        TOKEN_USAGE["calls"] += 1
        if usage:
            TOKEN_USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            TOKEN_USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        else:
            # API usage döndürmediyse kaba tahmin (yaklaşık 4 karakter/token)
            TOKEN_USAGE["usage_missing"] += 1
            TOKEN_USAGE["prompt_tokens"] += max(1, len(str(payload["messages"])) // 4)
            TOKEN_USAGE["completion_tokens"] += max(1, len(content or "") // 4)
        return content or "", usage
    logger.warning("LLM isteği %d denemede tamamlanamadı", LLM_MAX_RETRIES)
    RUN_STATS["failed_requests"] += 1
    return None, {}


def tag_with_llm(title: str) -> dict:
    """Tek başlığı etiketler (toplu yolun yedek planı olarak da kullanılır)."""
    # Hata durumunda "_error" anahtarı eklenir; çağıran taraf bu sonucu önbelleğe almamalıdır.
    empty = {"kategoriler": [], "aktorler": [], "_error": True}
    title = (title or "").strip()
    if not title:
        return empty
    content, _ = _post(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": title[:MAX_TITLE_CHARS]},
        ],
        SINGLE_MAX_TOKENS,
    )
    if content is None:
        return empty
    data = _extract_json(content)
    if data is None:
        RUN_STATS["bad_json"] += 1
        logger.warning("LLM geçersiz JSON döndürdü: %s", content[:120])
        return empty
    return _clean_result(data)


def tag_batch_with_llm(titles: list[str]) -> list[dict]:
    """Başlıkları tek istekte etiketler; her başlık için sonuç listesi döndürür."""
    empty = {"kategoriler": [], "aktorler": [], "_error": True}
    results = [dict(empty) for _ in titles]
    cleaned_titles = [(title or "").strip()[:MAX_TITLE_CHARS] for title in titles]
    if not cleaned_titles or not any(cleaned_titles):
        return results
    prompt = "\n".join(f"{index}) {title}" for index, title in enumerate(cleaned_titles, start=1))
    content, _ = _post(
        [
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        BATCH_MAX_TOKENS,
    )
    if content is None:
        return results
    data = _extract_json(content)
    if data is None:
        RUN_STATS["bad_json"] += 1
        logger.warning("Toplu yanıt geçersiz JSON: %s", content[:120])
        return results
    for item in data.get("sonuclar") or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("id")) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(results):
            continue
        results[index] = _clean_result(item)
    return results


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


def _estimate_cost() -> float:
    return (TOKEN_USAGE["prompt_tokens"] / 1_000_000 * COST_INPUT_PER_MTOK
            + TOKEN_USAGE["completion_tokens"] / 1_000_000 * COST_OUTPUT_PER_MTOK)


def _store(row: dict, result: dict) -> None:
    row["kategoriler"] = json.dumps(result.get("kategoriler", []), ensure_ascii=False)
    row["aktorler"] = json.dumps(result.get("aktorler", []), ensure_ascii=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agora - haber başlığı etiketleyici")
    parser.add_argument("--limit", type=int, default=0, help="En fazla kaç yeni başlık etiketlensin (0 = tümü)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="İstek başına başlık sayısı")
    args = parser.parse_args(argv)

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

    TOKEN_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usage_missing": 0})
    RUN_STATS.update({"retries": 0, "failed_requests": 0, "bad_json": 0})
    started = time.monotonic()

    limit = max(0, args.limit)
    hits = 0
    pending: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        key = _cache_key(row)
        if key in cache:
            _store(row, cache[key])
            hits += 1
            continue
        if limit and len(pending) >= limit:
            row["kategoriler"] = "[]"
            row["aktorler"] = "[]"
            continue
        pending.append((index, key))

    batch_size = max(1, args.batch_size)
    batch_calls = 0
    single_calls = 0
    failures = 0
    logger.info("Etiketlenecek: %d başlık (%d istek planlandı, %d başlık/istek)",
                len(pending), math.ceil(len(pending) / batch_size), batch_size)

    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset:offset + batch_size]
        titles = [rows[index].get("title", "") for index, _ in chunk]
        if batch_size == 1:
            # Varsayılan yol: başlık başına tek istek
            single_calls += 1
            results = [tag_with_llm(titles[0])]
        else:
            batch_calls += 1
            results = tag_batch_with_llm(titles)
        for (index, key), result in zip(chunk, results):
            if result.get("_error"):
                # Toplu yanıtta eksik kalan başlık için tekil istek
                result = tag_with_llm(rows[index].get("title", ""))
                single_calls += 1
            if result.get("_error"):
                failures += 1
                rows[index]["kategoriler"] = "[]"
                rows[index]["aktorler"] = "[]"
                continue
            cache[key] = result
            _store(rows[index], result)
        _save_cache(cache_path, cache)
        logger.info("Etiketleniyor: %d/%d", min(offset + batch_size, len(pending)), len(pending))

    _save_cache(cache_path, cache)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - started
    logger.info("Önbellek: %d isabet | istek: tekil %d, toplu %d | hata: %d",
                hits, single_calls, batch_calls, failures)
    logger.info("Hata/deneme: yeniden deneme %d, başarısız istek %d, geçersiz JSON %d",
                RUN_STATS["retries"], RUN_STATS["failed_requests"], RUN_STATS["bad_json"])
    logger.info("Süre: %.1f sn | başarılı istek: %d | token: giriş %d, çıkış %d | tahmini maliyet: $%.5f",
                elapsed, TOKEN_USAGE["calls"], TOKEN_USAGE["prompt_tokens"],
                TOKEN_USAGE["completion_tokens"], _estimate_cost())
    if TOKEN_USAGE["usage_missing"]:
        logger.warning("%d istekte API token bilgisi dönmedi, değerler tahminidir",
                       TOKEN_USAGE["usage_missing"])
    logger.info("Etiketleme tamamlandı: %s (%d satır)", output_path, len(rows))


if __name__ == "__main__":
    main()
