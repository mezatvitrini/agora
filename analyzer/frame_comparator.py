"""Agora - çerçeveleme karşılaştırması.

Aynı olayı işleyen haberleri gruplar ve her grup için LLM'e "bu haberler aynı olayı
farklı kaynaklarda nasıl sunuyor?" sorusunu sorar. Sonuç
data/processed/frame_comparison.json dosyasına yazılır.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from datetime import date
from itertools import combinations
from pathlib import Path
import requests
from dateutil import parser as date_parser
from config import PROCESSED_CSV

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.frame")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from analyzer.category_tagger import _api_key, _extract_json
except ImportError:  # doğrudan script olarak çalıştırıldığında
    sys.path.insert(0, str(PROJECT_ROOT))
    from analyzer.category_tagger import _api_key, _extract_json

FRAME_OUTPUT = "data/processed/frame_comparison.json"
FRAME_CACHE = "data/processed/frame_cache.json"

LLM_API_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = "deepseek-chat"
LLM_TIMEOUT = 90
LLM_MAX_RETRIES = 3
LLM_MAX_TOKENS = 4000
PREVIEW_LEN = 8000
MAX_TITLE_CHARS = 300

# deepseek-chat fiyatları (USD / 1M token; önbellek isabetsiz giriş ve çıkış)
COST_INPUT_PER_MTOK = 0.27
COST_OUTPUT_PER_MTOK = 1.10

TITLE_SIMILARITY_THRESHOLD = 0.45
MIN_CONTAINMENT_TOKENS = 3
ACTOR_MIN_SHARED = 2
ACTOR_PARTIAL_TITLE_OVERLAP = 0.2
ACTOR_DAY_WINDOW_DAYS = 2

MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 6
MAX_GROUPS = 40
CACHE_SAVE_EVERY = 5
PROMPT_VERSION = "v1"

# İstek/token sayaçları (main() başında sıfırlanır)
TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usage_missing": 0}

# Başlık karşılaştırmasında anlam taşımayan kelimeler.
STOPWORDS = {
    "ve", "ile", "bir", "bu", "da", "de", "ki", "mi", "mı", "mu", "mü", "ne", "o",
    "olan", "olarak", "için", "gibi", "daha", "çok", "en", "sonra", "önce", "yeni",
    "göre", "ama", "ancak", "ise", "ya", "veya", "hem", "her", "şu", "tüm", "bütün",
    "üzerine", "dair", "karşı", "yok", "var", "değil", "oldu", "olduğu", "dedi",
    "söyledi", "belirtti", "açıkladı", "yaptı", "etti", "ettiği", "aldı", "verdi",
    "haber", "haberleri", "son", "dakika", "canlı", "video", "foto", "galeri",
}

# Tek başına olay ayırt etmeyen, her haberde geçebilen aktörler.
GENERIC_ACTORS = {
    "turkiye", "abd", "ab", "avrupa birligi", "birlesmis milletler", "bm", "nato",
    "turk", "hukumet", "cumhurbaskani", "bakan", "bakanlik", "meclis", "tbmm",
    "genelkurmay", "polis", "ordu", "valilik", "emniyet", "savcilik", "mahkeme",
    "dunya", "avrupa", "orta dogu", "kamu", "devlet", "millet", "halk", "ulkemiz",
    # Yer adları aktör olarak listelense de olayı tek başına ayırt etmez.
    "istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya", "gaziantep",
    "sanliurfa", "mersin", "diyarbakir", "kayseri", "eskisehir", "samsun", "denizli",
    "malatya", "trabzon", "erzurum", "van", "batman", "hatay", "kocaeli", "tekirdag",
}

_TR_FOLD = str.maketrans({
    "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u", "ô": "o", "é": "e",
})

SYSTEM_PROMPT = (
    "Sen bir Türkçe haber çözümleyicisisin. Sana aynı olayı işlediği düşünülen haberler, "
    "kaynaklarıyla birlikte verilir. Görevin: bu haberler aynı olayı farklı kaynaklarda "
    "nasıl sunuyor? Hangi kelimeler seçilmiş, kim aktif özne, hangi bilgi verilmiş, "
    "hangi bilgi atlanmış?\n"
    "Yalnızca geçerli JSON döndür, başka açıklama yazma. Yalnızca verilen başlıklardaki "
    "bilgilere dayan; bilgi uydurma.\n"
    "JSON şeması:\n"
    "{\n"
    '  "olay": "olayın tek cümlelik nötr özeti",\n'
    '  "kaynak_analizleri": [\n'
    '    {"kaynak": "...", "baslik": "...", "secili_kelimeler": ["..."], '
    '"aktif_ozne": "...", "verilen_bilgiler": ["..."], '
    '"atlanan_bilgiler": ["..."], "cerceve_notu": "..."}\n'
    "  ],\n"
    '  "ortak_bilgiler": ["..."],\n'
    '  "ayrisan_noktalar": ["..."],\n'
    '  "cerceve_farklari": [{"boyut": "kelime seçimi|aktif özne|bilgi seçimi", "aciklama": "..."}],\n'
    '  "ozet": "kaynaklar arasındaki çerçeve farkını anlatan kısa değerlendirme"\n'
    "}\n"
    "kaynak_analizleri dizisinde verilen her habere bir kayıt olmalı. Haberler gerçekten "
    "farklı olayları anlatıyorsa bunu kayıtların cerceve_notu ve ozet alanında açıkça belirt."
)


# ---------------------------------------------------------------- yardımcılar

def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _fold(text: str) -> str:
    return text.lower().replace("\u0307", "").translate(_TR_FOLD)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", _fold(text)).split())


def _title_tokens(title: str) -> frozenset[str]:
    return frozenset(w for w in _normalize(title).split() if w not in STOPWORDS)


def _actor_keys(actors: list[str]) -> frozenset[str]:
    return frozenset(
        key for key in (_normalize(a) for a in actors)
        if key and key not in GENERIC_ACTORS
    )


def _json_list(value: str | list | None) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(v).strip() for v in data if str(v).strip()] if isinstance(data, list) else []


def _parse_day(value: str) -> date | None:
    if not value:
        return None
    try:
        return date_parser.parse(value).date()
    except (ValueError, OverflowError, TypeError):
        return None


def _day_gap(a: date | None, b: date | None) -> int | None:
    if a is None or b is None:
        return None
    return abs((a - b).days)


# ---------------------------------------------------------------- gruplama

def _build_items(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        title = (row.get("title") or "").strip()
        if not title:
            continue
        actors = _json_list(row.get("aktorler"))
        items.append({
            "source": (row.get("source") or "?").strip(),
            "title": title,
            "link": (row.get("link") or "").strip(),
            "published": (row.get("published") or "").strip(),
            "day": _parse_day(row.get("published") or ""),
            "dedup_hash": (row.get("dedup_hash") or "").strip(),
            "categories": _json_list(row.get("kategoriler")),
            "actors": actors,
            "tokens": _title_tokens(title),
            "actor_keys": _actor_keys(actors),
        })
    return items


def _numbers_conflict(a: frozenset[str], b: frozenset[str]) -> bool:
    nums_a = {w for w in a if w.isdigit()}
    nums_b = {w for w in b if w.isdigit()}
    if not nums_a or not nums_b:
        return False
    return not (nums_a <= nums_b or nums_b <= nums_a)


def _prefix_match(a: str, b: str) -> bool:
    """Türkçe ekler yüzünden aynı kökten gelen kelimeleri eşler (operasyonu ~ operasyonunda)."""
    if a == b:
        return True
    if a.isdigit() or b.isdigit() or min(len(a), len(b)) < 5:
        return False
    return a.startswith(b) or b.startswith(a)


def _title_similarity(tokens_a: frozenset[str], tokens_b: frozenset[str]) -> float:
    matched = 0
    for a in tokens_a:
        if any(_prefix_match(a, b) for b in tokens_b):
            matched += 1
    union = len(tokens_a) + len(tokens_b) - matched
    return min(matched / union, 1.0) if union else 0.0


def _link_reasons(a: dict, b: dict) -> list[str]:
    tokens_a, tokens_b = a["tokens"], b["tokens"]
    union = tokens_a | tokens_b
    jaccard = _title_similarity(tokens_a, tokens_b) if union else 0.0

    title_link = False
    if union and not _numbers_conflict(tokens_a, tokens_b):
        if jaccard >= TITLE_SIMILARITY_THRESHOLD:
            title_link = True
        else:
            shorter, longer = (
                (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
            )
            title_link = len(shorter) >= MIN_CONTAINMENT_TOKENS and shorter <= longer

    actor_link = False
    shared_actors = a["actor_keys"] & b["actor_keys"]
    gap = _day_gap(a["day"], b["day"])
    if shared_actors and (gap is None or gap <= ACTOR_DAY_WINDOW_DAYS):
        if len(shared_actors) >= ACTOR_MIN_SHARED:
            actor_link = True
        elif jaccard >= ACTOR_PARTIAL_TITLE_OVERLAP:
            actor_link = True

    reasons = []
    if title_link:
        reasons.append("baslik")
    if actor_link:
        reasons.append("aktor")
    return reasons


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        while self._parent[index] != index:
            self._parent[index] = self._parent[self._parent[index]]
            index = self._parent[index]
        return index

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def _trim_group(items: list[dict]) -> tuple[list[dict], bool]:
    if len(items) <= MAX_GROUP_SIZE:
        return items, False
    keep: list[dict] = []
    seen_sources: set[str] = set()
    for item in items:
        if item["source"] not in seen_sources:
            keep.append(item)
            seen_sources.add(item["source"])
        if len(keep) >= MAX_GROUP_SIZE:
            break
    for item in items:
        if len(keep) >= MAX_GROUP_SIZE:
            break
        if item not in keep:
            keep.append(item)
    return keep[:MAX_GROUP_SIZE], True


def group_by_event(items: list[dict], min_sources: int = 2) -> list[dict]:
    """Aynı olayı işleyen haberleri başlık benzerliği veya ortak aktörlere göre gruplar."""
    uf = _UnionFind(len(items))
    for i, j in combinations(range(len(items)), 2):
        if _link_reasons(items[i], items[j]):
            uf.union(i, j)

    buckets: dict[int, list[int]] = {}
    for index in range(len(items)):
        buckets.setdefault(uf.find(index), []).append(index)

    groups = []
    for indexes in buckets.values():
        if len(indexes) < MIN_GROUP_SIZE:
            continue
        members = [items[i] for i in indexes]
        if len({m["source"] for m in members}) < min_sources:
            continue
        reasons: set[str] = set()
        for i, j in combinations(indexes, 2):
            reasons.update(_link_reasons(items[i], items[j]))
        members.sort(key=lambda m: (m["published"] or "", m["source"], m["title"]))
        members, trimmed = _trim_group(members)
        groups.append({
            "eslesme": "+".join(sorted(reasons)) or "baslik",
            "kaynaklar": sorted({m["source"] for m in members}),
            "haberler": members,
            "kisaltildi": trimmed,
        })
    groups.sort(key=lambda g: (-len(g["haberler"]), -len(g["kaynaklar"]), g["haberler"][0]["title"]))
    return groups


# ---------------------------------------------------------------- LLM

def _build_user_prompt(members: list[dict]) -> str:
    lines = [
        f"Aşağıda aynı olayı işlediği değerlendirilen {len(members)} haber var "
        f"({len({m['source'] for m in members})} farklı kaynak).",
        "",
    ]
    for number, item in enumerate(members, start=1):
        categories = ", ".join(item["categories"])[:MAX_TITLE_CHARS] or "—"
        actors = ", ".join(item["actors"])[:MAX_TITLE_CHARS] or "—"
        published = item["published"] or "—"
        lines.append(f"{number}) Kaynak: {item['source']}")
        lines.append(f"   Yayın: {published}")
        lines.append(f"   Başlık: {item['title'][:MAX_TITLE_CHARS]}")
        lines.append(f"   Kategoriler: {categories}")
        lines.append(f"   Aktörler: {actors}")
        lines.append("")
    lines.append(
        "Soruları her kaynak için ayrı yanıtla: hangi kelimeler seçilmiş, kim aktif özne, "
        "hangi bilgi verilmiş, hangi bilgi atlanmış? Yalnızca JSON döndür."
    )
    return "\n".join(lines)[:PREVIEW_LEN]


def _estimate_cost() -> float:
    return (TOKEN_USAGE["prompt_tokens"] / 1_000_000 * COST_INPUT_PER_MTOK
            + TOKEN_USAGE["completion_tokens"] / 1_000_000 * COST_OUTPUT_PER_MTOK)


def _record_usage(body: dict, content: str, prompt_chars: int) -> None:
    usage = body.get("usage") or {}
    TOKEN_USAGE["calls"] += 1
    if usage:
        TOKEN_USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        TOKEN_USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    else:
        # API usage döndürmediyse kaba tahmin (yaklaşık 4 karakter/token)
        TOKEN_USAGE["usage_missing"] += 1
        TOKEN_USAGE["prompt_tokens"] += max(1, prompt_chars // 4)
        TOKEN_USAGE["completion_tokens"] += max(1, len(content or "") // 4)


def _call_llm(members: list[dict]) -> tuple[dict | None, str | None]:
    """(veri, hata_nedeni) döndürür. hata_nedeni: yanit_kesildi / gecersiz_json / istek_hatasi."""
    key = _api_key()
    if not key:
        logger.error("DEEPSEEK_API_KEY bulunamadı (.env veya ortam değişkeni)")
        return None, "api_anahtari_yok"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(members)},
        ],
        "temperature": 0,
        "max_tokens": LLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    prompt_chars = sum(len(message["content"]) for message in payload["messages"])
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
            return None, "istek_hatasi"
        try:
            body = resp.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, ValueError, TypeError):
            logger.warning("LLM yanıtı beklenen biçimde değil")
            return None, "istek_hatasi"
        _record_usage(body, content or "", prompt_chars)
        data = _extract_json(content or "")
        if data is None:
            cut = finish_reason == "length" or not (content or "").rstrip().endswith("}")
            if cut:
                logger.warning("Yanıt kesildi (max_tokens=%d, %d karakter): grup %d haber içeriyor",
                               LLM_MAX_TOKENS, len(content or ""), len(members))
                return None, "yanit_kesildi"
            logger.warning("LLM geçersiz JSON döndürdü: %s", (content or "")[:160])
            return None, "gecersiz_json"
        return data, None
    logger.warning("LLM isteği %d denemede tamamlanamadı", LLM_MAX_RETRIES)
    return None, "istek_hatasi"


def _clean_comparison(data: dict, members: list[dict]) -> dict:
    def as_text(value) -> str:
        return str(value).strip()[:1200] if value not in (None, "") else ""

    def as_list(value, limit: int = 8) -> list[str]:
        source = value if isinstance(value, (list, tuple)) else ([value] if value else [])
        result: list[str] = []
        for raw in source:
            text = str(raw).strip()[:400]
            if text and text not in result:
                result.append(text)
        return result[:limit]

    by_title = {member["title"].strip().lower(): member for member in members}
    by_source: dict[str, list[dict]] = {}
    for member in members:
        by_source.setdefault(_fold(member["source"]), []).append(member)

    analyses: list[dict] = []
    used: set[int] = set()
    for raw in data.get("kaynak_analizleri") or []:
        if not isinstance(raw, dict):
            continue
        title = as_text(raw.get("baslik"))
        source = as_text(raw.get("kaynak"))
        member = by_title.get(title.strip().lower())
        if member is None:
            candidates = by_source.get(_fold(source), [])
            free = [c for c in candidates if id(c) not in used]
            member = free[0] if free else None
        if member is None:
            continue
        used.add(id(member))
        analyses.append({
            "kaynak": member["source"],
            "baslik": member["title"],
            "secili_kelimeler": as_list(raw.get("secili_kelimeler")),
            "aktif_ozne": as_text(raw.get("aktif_ozne")),
            "verilen_bilgiler": as_list(raw.get("verilen_bilgiler")),
            "atlanan_bilgiler": as_list(raw.get("atlanan_bilgiler")),
            "cerceve_notu": as_text(raw.get("cerceve_notu")),
        })
    # LLM bir haberi atladıysa boş kayıtla tamamla: her haber sonuçta görünmeli.
    for member in members:
        if id(member) not in used:
            analyses.append({
                "kaynak": member["source"],
                "baslik": member["title"],
                "secili_kelimeler": [],
                "aktif_ozne": "",
                "verilen_bilgiler": [],
                "atlanan_bilgiler": [],
                "cerceve_notu": "",
                "_llm_atladi": True,
            })

    frames = []
    for raw in data.get("cerceve_farklari") or []:
        if isinstance(raw, dict):
            frames.append({"boyut": as_text(raw.get("boyut")), "aciklama": as_text(raw.get("aciklama"))})
        elif raw:
            frames.append({"boyut": "", "aciklama": as_text(raw)})

    return {
        "olay": as_text(data.get("olay")),
        "kaynak_analizleri": analyses,
        "ortak_bilgiler": as_list(data.get("ortak_bilgiler")),
        "ayrisan_noktalar": as_list(data.get("ayrisan_noktalar")),
        "cerceve_farklari": [f for f in frames if f["aciklama"] or f["boyut"]][:8],
        "ozet": as_text(data.get("ozet")),
    }


# ---------------------------------------------------------------- önbellek

def _cache_key(members: list[dict]) -> str:
    raw = "|".join(sorted(m["dedup_hash"] or m["title"] for m in members))
    return hashlib.sha256(f"{PROMPT_VERSION}|{LLM_MODEL}|{raw}".encode("utf-8")).hexdigest()[:32]


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Önbellek okunamadı, sıfırdan başlanıyor: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- ana akış

def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("Girdi bulunamadı: %s", path)
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "title" not in rows[0]:
        logger.error("Girdide 'title' sütunu yok: %s", path)
        return []
    return rows


def _public_group(group: dict, comparison: dict | None, error: str | None = None) -> dict:
    result = {
        "eslesme": group["eslesme"],
        "kaynaklar": group["kaynaklar"],
        "haberler": [
            {
                "kaynak": item["source"],
                "baslik": item["title"],
                "link": item["link"],
                "yayin": item["published"],
            }
            for item in group["haberler"]
        ],
        "karsilastirma": comparison,
    }
    if group.get("kisaltildi"):
        result["not"] = f"Grup {MAX_GROUP_SIZE} haberle sınırlandı."
    if error:
        result["hata"] = error
        result["basarisiz"] = True
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agora - haberlerde çerçeveleme karşılaştırması")
    parser.add_argument("--input", default=str(_path(PROCESSED_CSV)), help="etiketlenmiş haber CSV'si")
    parser.add_argument("--output", default=str(_path(FRAME_OUTPUT)), help="çıktı JSON dosyası")
    parser.add_argument("--cache", default=str(_path(FRAME_CACHE)), help="LLM önbelleği")
    parser.add_argument("--max-groups", type=int, default=MAX_GROUPS, help="en fazla kaç grup karşılaştırılsın")
    parser.add_argument("--min-sources", type=int, default=2, help="grup için gereken en az kaynak sayısı")
    parser.add_argument("--force", action="store_true", help="önbelleği yok say, LLM'i yeniden çağır")
    args = parser.parse_args(argv)

    items = _build_items(_read_rows(Path(args.input)))
    if not items:
        logger.error("Karşılaştırılacak haber bulunamadı")
        return

    groups = group_by_event(items, min_sources=args.min_sources)
    logger.info("%d haber, %d olay grubu bulundu", len(items), len(groups))
    selected = groups[:max(0, args.max_groups)]
    if len(groups) > len(selected):
        logger.info("Grup sınırı: %d/%d grup karşılaştırılacak", len(selected), len(groups))

    has_key = bool(_api_key())
    if not has_key:
        logger.error("DEEPSEEK_API_KEY yok; yalnızca gruplama yazılacak, karşılaştırma atlanıyor")

    cache_path, output_path = Path(args.cache), Path(args.output)
    cache = _load_cache(cache_path) if has_key else {}
    TOKEN_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usage_missing": 0})
    started = time.monotonic()
    hits = calls = failures = unsaved = 0
    failures_list: list[tuple[int, str]] = []
    results = []
    for number, group in enumerate(selected, start=1):
        comparison, error = None, None
        key = _cache_key(group["haberler"])
        if not has_key:
            error = "api_anahtari_yok"
        elif not args.force and key in cache:
            comparison = cache[key]
            hits += 1
        else:
            logger.info("Grup %d/%d karşılaştırılıyor (%d haber)", number, len(selected), len(group["haberler"]))
            raw, reason = _call_llm(group["haberler"])
            calls += 1
            if raw is None:
                failures += 1
                error = reason or "llm_hatasi"
                failures_list.append((number, error))
                logger.warning("Grup %d BAŞARISIZ (%s) — grubun karşılaştırması boş kalacak", number, error)
            else:
                comparison = _clean_comparison(raw, group["haberler"])
                cache[key] = comparison
                unsaved += 1
                if unsaved >= CACHE_SAVE_EVERY:
                    _save_json(cache_path, cache)
                    unsaved = 0
        entry = _public_group(group, comparison, error)
        entry = {"grup_id": number, **entry}
        results.append(entry)
    if has_key:
        _save_json(cache_path, cache)

    elapsed = time.monotonic() - started
    payload = {
        "uretim_zamani": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "girdi": str(args.input),
        "model": LLM_MODEL,
        "durum": "tamam" if has_key and not failures else ("api_anahtari_yok" if not has_key else "kismi"),
        "haber_sayisi": len(items),
        "grup_sayisi": len(selected),
        "toplam_grup": len(groups),
        "istatistik": {
            "llm_cagrisi": TOKEN_USAGE["calls"],
            "onbellek_isabeti": hits,
            "giris_token": TOKEN_USAGE["prompt_tokens"],
            "cikis_token": TOKEN_USAGE["completion_tokens"],
            "tahmini_maliyet_usd": round(_estimate_cost(), 5),
            "sure_sn": round(elapsed, 1),
            "basarisiz_gruplar": [{"grup_id": gid, "neden": reason} for gid, reason in failures_list],
        },
        "gruplar": results,
    }
    _save_json(output_path, payload)
    logger.info("Önbellek: %d isabet, %d LLM çağrısı, %d hata", hits, calls, failures)
    if failures_list:
        logger.warning("Başarısız gruplar: %s",
                       ", ".join(f"{gid} ({reason})" for gid, reason in failures_list))
    logger.info("Süre: %.1f sn | token: giriş %d, çıkış %d | tahmini maliyet: $%.5f",
                elapsed, TOKEN_USAGE["prompt_tokens"], TOKEN_USAGE["completion_tokens"],
                _estimate_cost())
    if TOKEN_USAGE["usage_missing"]:
        logger.warning("%d istekte API token bilgisi dönmedi, değerler tahminidir",
                       TOKEN_USAGE["usage_missing"])
    logger.info("Çıktı yazıldı: %s (%d grup)", output_path, len(results))


if __name__ == "__main__":
    main()
