"""Alıntı doğrulama — tamamen yerel, LLM çağrısı yapmaz.

frame_comparison_embedding.json içindeki her kaynak analizinin "kisa_alinti" alanını,
o kümedeki başlıklarda arar. Önce aynı kaynağın başlığı denenir, yoksa kümedeki diğer
başlıklara bakılır.

Sonuçlar:
- "birebir": normalize edilmiş alıntı, normalize edilmiş başlığın alt dizesidir.
- "yakin": benzerlik >= YAKIN_ESIK (0.85). Benzerlik = max(difflib oranı, alıntı
  kelimelerinin başlıkta görünme oranı).
- "yok": eşleşme bulunamadı.
- "belirsiz": alıntı alanı boş veya model "belirsiz" yazmış.
"""
from __future__ import annotations
import argparse
import difflib
import json
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.verify")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_JSON = "data/processed/frame_comparison_embedding.json"
YAKIN_ESIK = 0.85
BELIRSIZ = "belirsiz"

_TR_FOLD = str.maketrans({
    "ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c",
    "â": "a", "î": "i", "û": "u", "ô": "o", "é": "e", "İ": "i",
})

KAYNAK_TIPI_ETIKET = {
    "devlet_baglantili": "devlet_kontrollu",
    "ozel": "ozel",
    "ajans": "ajans",
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize(text: str) -> str:
    """Küçük harf + Türkçe katlama + noktalama temizliği + boşluk sadeleştirme."""
    folded = str(text or "").lower().replace("\u0307", "").translate(_TR_FOLD)
    temiz = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in folded)
    return " ".join(temiz.split())


def _partial_ratio(quote: str, headline: str) -> float:
    """Alıntı, başlığın bir parçasına ne kadar benziyor (kayan pencere)."""
    if not quote or not headline:
        return 0.0
    if len(quote) >= len(headline):
        return difflib.SequenceMatcher(None, quote, headline).ratio()
    best = 0.0
    son = len(headline) - len(quote) + 1
    for start in range(son):
        window = headline[start:start + len(quote)]
        best = max(best, difflib.SequenceMatcher(None, quote, window).ratio())
    return best


def similarity(quote: str, headline: str) -> float:
    if not quote or not headline:
        return 0.0
    ratio = difflib.SequenceMatcher(None, quote, headline).ratio()
    partial = _partial_ratio(quote, headline)
    quote_tokens, headline_tokens = quote.split(), set(headline.split())
    coverage = (sum(1 for token in quote_tokens if token in headline_tokens) / len(quote_tokens)
                if quote_tokens else 0.0)
    return max(ratio, partial, coverage)


def verify_quote(quote: str, candidates: list[str], threshold: float = YAKIN_ESIK) -> tuple[str, str]:
    """(sonuç, eşleşen başlık) döndürür: birebir / yakin / yok / belirsiz."""
    normalized_quote = normalize(quote)
    if not normalized_quote or normalized_quote == BELIRSIZ:
        return "belirsiz", ""
    best_score = 0.0
    best_candidate = ""
    for candidate in candidates:
        normalized_headline = normalize(candidate)
        if not normalized_headline:
            continue
        if normalized_quote in normalized_headline:
            return "birebir", candidate
        score = similarity(normalized_quote, normalized_headline)
        if score > best_score:
            best_score, best_candidate = score, candidate
    if best_score >= threshold:
        return "yakin", best_candidate
    return "yok", best_candidate


def load_source_types(path: str | Path | None = None) -> dict[str, str]:
    """Kaynak -> kaynak_tipi eşlemesi (etiketlenmiş CSV'den; yerel dosya okuma)."""
    from config import PROCESSED_CSV

    target = Path(path) if path else _path(PROCESSED_CSV)
    types: dict[str, str] = {}
    if not target.exists():
        logger.warning("Kaynak tipi için CSV bulunamadı: %s", target)
        return types
    import csv as _csv
    with target.open(newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            source = (row.get("source") or "").strip()
            kind = (row.get("kaynak_tipi") or "").strip()
            if source and kind and source not in types:
                types[source] = kind
    return types


def verify_cluster(cluster: dict, source_types: dict[str, str] | None = None,
                   threshold: float = YAKIN_ESIK) -> list[dict]:
    """Bir kümedeki her kaynak analizi için (alıntı, sonuç, eşleşen başlık) kaydı üretir."""
    source_types = source_types or {}
    headlines = [h.get("baslik", "") for h in cluster.get("haberler") or []]
    by_source: dict[str, list[str]] = {}
    for head in cluster.get("haberler") or []:
        by_source.setdefault(head.get("kaynak", ""), []).append(head.get("baslik", ""))

    records = []
    for analiz in (cluster.get("karsilastirma") or {}).get("kaynak_analizleri") or []:
        kaynak = analiz.get("kaynak", "")
        quote = analiz.get("kisa_alinti", "")
        # 1) aynı kaynağın başlığı, 2) kümedeki diğer başlıklar
        same_source = by_source.get(kaynak, [])
        others = [h for h in headlines if h not in same_source]
        sonuc, eslesen = verify_quote(quote, same_source, threshold)
        if sonuc == "yok":
            sonuc, eslesen = verify_quote(quote, others, threshold)
        kaynak_tipi = KAYNAK_TIPI_ETIKET.get(source_types.get(kaynak, ""), source_types.get(kaynak, ""))
        records.append({
            "kume_id": cluster.get("kume_id", ""),
            "kaynak": kaynak,
            "ulke": analiz.get("ulke", ""),
            "kaynak_tipi": kaynak_tipi,
            "ton": analiz.get("ton", ""),
            "alinti": quote,
            "sonuc": sonuc,
            "eslesen_baslik": eslesen,
            "kaynak_basligi": (same_source[0] if same_source else ""),
        })
    return records


def apply_verification(data: dict, threshold: float = YAKIN_ESIK,
                       source_types: dict[str, str] | None = None) -> dict:
    """Sonuçları JSON verisine 'alinti_dogrulandi' alanı olarak işler; sayaçları döndürür."""
    source_types = source_types if source_types is not None else load_source_types()
    sayac = Counter()
    detay = {}
    for cluster in data.get("kumeler") or []:
        records = verify_cluster(cluster, source_types, threshold)
        detay[cluster.get("kume_id", "")] = records
        analyses = (cluster.get("karsilastirma") or {}).get("kaynak_analizleri") or []
        for analiz, record in zip(analyses, records):
            analiz["alinti_dogrulandi"] = record["sonuc"]
            sayac[record["sonuc"]] += 1
    return {"sayac": sayac, "detay": detay}


def source_type_ton_table(records: list[dict]) -> dict[tuple[str, str], int]:
    table: Counter = Counter()
    for record in records:
        tip = record["kaynak_tipi"] or "—"
        ton = record["ton"] or "—"
        table[(tip, ton)] += 1
    return dict(table)


def print_report(data: dict, sonuc: dict, threshold: float = YAKIN_ESIK) -> None:
    sayac = sonuc["sayac"]
    kayitlar = [r for records in sonuc["detay"].values() for r in records]
    print("\n=== alıntı doğrulama ===")
    print(f"  toplam analiz: {len(kayitlar)} | birebir: {sayac['birebir']} | "
          f"yakin ({threshold}): {sayac['yakin']} | yok: {sayac['yok']} | belirsiz: {sayac['belirsiz']}")

    print("\n=== 'yok' olan alıntılar ===")
    bulunamayan = [r for r in kayitlar if r["sonuc"] == "yok"]
    if not bulunamayan:
        print("  yok")
    for r in bulunamayan:
        print(f"  [{r['kume_id']} · {r['kaynak']}] alıntı: {r['alinti'][:90]!r}")
        print(f"      kaynağın başlığı: {r['kaynak_basligi'][:100]!r}")

    print("\n=== kaynak_tipi x ton (analiz sayısı) ===")
    table = source_type_ton_table(kayitlar)
    tipler = sorted({tip for tip, _ in table})
    tonlar = sorted({ton for _, ton in table})
    print(f"{'kaynak_tipi':18}" + "".join(f"{ton:>10}" for ton in tonlar) + f"{'toplam':>10}")
    for tip in tipler:
        satir = [table.get((tip, ton), 0) for ton in tonlar]
        print(f"{tip:18}" + "".join(f"{sayi:>10}" for sayi in satir) + f"{sum(satir):>10}")
    print("\n  Not: örneklem küçük, genelleme yapılamaz.")

    for cluster in data.get("kumeler") or []:
        if (cluster.get("karsilastirma") or {}).get("kume_uyumsuz"):
            print(f"\n=== uyumsuz küme {cluster.get('kume_id')} başlıkları ===")
            for head in cluster.get("haberler") or []:
                print(f"  - ({head.get('kaynak')} | {head.get('ulke')}) {head.get('baslik')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agora - alıntı doğrulama (LLM çağrısı yok)")
    parser.add_argument("--input", default=str(_path(EMBEDDING_JSON)))
    parser.add_argument("--threshold", type=float, default=YAKIN_ESIK)
    parser.add_argument("--no-write", action="store_true", help="JSON'a alinti_dogrulandi yazma")
    args = parser.parse_args(argv)

    target = Path(args.input)
    if not target.exists():
        raise SystemExit(f"Girdi bulunamadı: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    sonuc = apply_verification(data, args.threshold)
    print_report(data, sonuc, args.threshold)

    if not args.no_write:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        logger.info("alinti_dogrulandi alanı yazıldı: %s", target)


if __name__ == "__main__":
    main()
