"""Agora - embedding tabanlı olay kümeleme.

Tamamen yereldir: LLM/API çağrısı yapmaz, frame comparator'ı çalıştırmaz.
Başlıklar orijinal dilinde (çevrilmeden) çok dilli bir modelle vektörleştirilir,
kosinüs benzerliği + agglomerative clustering ile olay kümeleri kurulur.

Mevcut aktör/kelime benzerliği yöntemi (analyzer/frame_comparator.py) olduğu gibi kalır;
bu modül onun yanında ikinci bir yöntem olarak durur.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from config import (
    CLUSTERS_CSV,
    CLUSTER_MAX_DAYS,
    CLUSTER_MAX_SIZE,
    CLUSTER_THRESHOLD,
    EMBEDDINGS_DIR,
    EMBEDDING_EXCLUDE_SOURCES,
    EMBEDDING_MODEL,
    EXCLUDED_CATEGORIES,
    FRAME_SKIP_SINGLE_COUNTRY,
    GAPS_CSV,
    PROCESSED_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.embedding")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIN_CLUSTER_SIZE = 2
GAPS_MIN_COUNTRIES = 3
THRESHOLD_GRID = (0.5, 0.6, 0.7)
ENCODING_BATCH_SIZE = 32

CLUSTER_COLUMNS = [
    "kume_id", "baslik", "kaynak", "ulke", "dil", "kaynak_tipi",
    "kume_boyutu", "ulke_sayisi", "tek_ulkeli",
]
GAP_COLUMNS = ["kume_id", "yazan_ulkeler", "yazmayan_ulkeler", "kapsama_orani"]


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _json_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(v).strip() for v in data if str(v).strip()] if isinstance(data, list) else []


def _parse_day(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(text).date()
    except (ValueError, OverflowError, TypeError):
        return None


# ---------------------------------------------------------------- veri hazırlığı

def load_rows(path: Path | None = None) -> list[dict]:
    source = Path(path) if path else _path(PROCESSED_CSV)
    if not source.exists():
        raise FileNotFoundError(f"Etiketlenmiş haber CSV'si bulunamadı: {source}")
    with source.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def prepare_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    """Kümelenecek satırları süzer: boş/spor kategorisi ve hariç kaynaklar dışarıda."""
    excluded_categories = set(EXCLUDED_CATEGORIES)
    excluded_sources = set(EMBEDDING_EXCLUDE_SOURCES)
    kept: list[dict] = []
    skipped: Counter = Counter()
    for row in rows:
        title = (row.get("title") or "").strip()
        source = (row.get("source") or "?").strip()
        if not title:
            skipped["baslik_yok"] += 1
            continue
        if source in excluded_sources:
            skipped["kaynak_hariç"] += 1
            continue
        categories = _json_list(row.get("kategoriler"))
        if not categories:
            skipped["kategori_bos"] += 1
            continue
        if excluded_categories & set(categories):
            skipped["kategori_hariç"] += 1
            continue
        kept.append({
            "title": title,
            "source": source,
            "ulke": (row.get("ulke") or "").strip(),
            "dil": (row.get("dil") or "").strip(),
            "kaynak_tipi": (row.get("kaynak_tipi") or "").strip(),
            "day": _parse_day(row.get("published")),
            "categories": categories,
        })
    return kept, skipped


# ---------------------------------------------------------------- embedding

def _cache_path(model_name: str = EMBEDDING_MODEL) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name) or "model"
    return _path(EMBEDDINGS_DIR) / f"{slug}.json"


def _embedding_key(title: str, model_name: str = EMBEDDING_MODEL) -> str:
    raw = f"{model_name}|{title.strip().lower()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Embedding önbelleği okunamadı, yeniden hesaplanacak: %s", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(path)


def load_model(model_name: str = EMBEDDING_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers kurulu değil. Kurulum: pip install sentence-transformers"
        ) from exc
    try:
        logger.info("Embedding modeli yükleniyor: %s", model_name)
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Embedding modeli yüklenemedi/indirilemedi ({model_name}): {exc}\n"
            "İnternet bağlantısını ve Hugging Face erişimini kontrol edin "
            "(gerekirse HF_HOME veya HTTPS_PROXY ayarlayın)."
        ) from exc


def embed_titles(titles: list[str], model=None, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Başlıkları orijinal dilinde vektörleştirir; önbellekte olanlar yeniden hesaplanmaz."""
    if not titles:
        return np.zeros((0, 0), dtype="float32")
    cache_path = _cache_path(model_name)
    cache = _load_cache(cache_path)
    keys = [_embedding_key(title, model_name) for title in titles]
    missing = [index for index, key in enumerate(keys) if key not in cache]
    if missing:
        model = model or load_model(model_name)
        texts = [titles[index] for index in missing]
        logger.info("%d/%d başlık için embedding hesaplanıyor (%d'lık gruplar, CPU)",
                    len(texts), len(titles), ENCODING_BATCH_SIZE)
        for start in range(0, len(texts), ENCODING_BATCH_SIZE):
            batch = texts[start:start + ENCODING_BATCH_SIZE]
            vectors = model.encode(
                batch,
                batch_size=ENCODING_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for offset, vector in enumerate(vectors):
                cache[keys[missing[start + offset]]] = [round(float(value), 6) for value in vector]
            # Her gruptan sonra kaydet: uzun süren iş kesilirse yeniden hesaplanmasın.
            _save_cache(cache_path, cache)
            logger.info("Embedding: %d/%d", min(start + ENCODING_BATCH_SIZE, len(texts)), len(texts))
    else:
        logger.info("Tüm embedding'ler önbellekten geldi (%d başlık)", len(titles))
    return np.asarray([cache[key] for key in keys], dtype="float32")


# ---------------------------------------------------------------- kümeleme

def _split_by_days(members: list[dict], max_days: int) -> list[list[dict]]:
    dated = [m for m in members if m["day"] is not None]
    undated = [m for m in members if m["day"] is None]
    dated.sort(key=lambda m: m["day"])
    parts: list[list[dict]] = []
    current: list[dict] = []
    for member in dated:
        if current and (member["day"] - current[0]["day"]).days > max_days:
            parts.append(current)
            current = []
        current.append(member)
    if current:
        parts.append(current)
    if undated:
        if parts:
            parts[-1].extend(undated)
        else:
            parts = [undated]
    return parts or [members]


def _split_by_size(members: list[dict], max_size: int) -> list[list[dict]]:
    if len(members) <= max_size:
        return [members]
    return [members[index:index + max_size] for index in range(0, len(members), max_size)]


def _cluster_from_vectors(
    rows: list[dict],
    vectors: np.ndarray,
    threshold: float = CLUSTER_THRESHOLD,
    max_days: int = CLUSTER_MAX_DAYS,
    max_size: int = CLUSTER_MAX_SIZE,
) -> list[list[dict]]:
    if len(rows) < MIN_CLUSTER_SIZE:
        return []
    from sklearn.cluster import AgglomerativeClustering

    # Kosinüs mesafesi = 1 - kosinüs benzerliği; vektörler normalize olduğu için eşik doğrudan dönüşür.
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=max(0.0, 1.0 - threshold),
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(vectors)

    buckets: dict[int, list[dict]] = defaultdict(list)
    for row, label in zip(rows, labels):
        buckets[int(label)].append(row)

    clusters: list[list[dict]] = []
    for members in buckets.values():
        for part in _split_by_days(members, max_days):
            for chunk in _split_by_size(part, max_size):
                if len(chunk) >= MIN_CLUSTER_SIZE:
                    clusters.append(chunk)
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]["title"]))
    return clusters


def cluster_rows(
    rows: list[dict],
    threshold: float = CLUSTER_THRESHOLD,
    max_days: int = CLUSTER_MAX_DAYS,
    max_size: int = CLUSTER_MAX_SIZE,
    model=None,
) -> list[list[dict]]:
    """Satırları embedding + agglomerative clustering ile olay kümelerine ayırır."""
    if len(rows) < MIN_CLUSTER_SIZE:
        return []
    vectors = embed_titles([row["title"] for row in rows], model=model)
    return _cluster_from_vectors(rows, vectors, threshold=threshold, max_days=max_days, max_size=max_size)


def _cluster_row(cluster_id: str, members: list[dict]) -> list[dict]:
    countries = {m["ulke"] for m in members if m["ulke"]}
    return [
        {
            "kume_id": cluster_id,
            "baslik": m["title"],
            "kaynak": m["source"],
            "ulke": m["ulke"],
            "dil": m["dil"],
            "kaynak_tipi": m["kaynak_tipi"],
            "kume_boyutu": len(members),
            "ulke_sayisi": len(countries),
            "tek_ulkeli": "evet" if len(countries) <= 1 else "hayir",
        }
        for m in members
    ]


def write_clusters(clusters: list[list[dict]], path: Path | None = None) -> Path:
    target = Path(path) if path else _path(CLUSTERS_CSV)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, members in enumerate(clusters, start=1):
        rows.extend(_cluster_row(f"K{index:03d}", members))
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLUSTER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Kümeler yazıldı: %s (%d küme, %d satır)", target, len(clusters), len(rows))
    return target


def tracked_countries(clusters: list[list[dict]]) -> list[str]:
    """İzlenen ülkeler: yalnızca veride gerçekten kaynağı olan ülkeler."""
    return sorted({m["ulke"] for members in clusters for m in members if m["ulke"]})


def write_gaps(clusters: list[list[dict]], path: Path | None = None,
               min_countries: int = GAPS_MIN_COUNTRIES) -> Path:
    target = Path(path) if path else _path(GAPS_CSV)
    target.parent.mkdir(parents=True, exist_ok=True)
    tracked = tracked_countries(clusters)
    rows = []
    for index, members in enumerate(clusters, start=1):
        present = {m["ulke"] for m in members if m["ulke"]}
        if len(present) < min_countries:
            continue
        missing = [country for country in tracked if country not in present]
        rows.append({
            "kume_id": f"K{index:03d}",
            "yazan_ulkeler": ";".join(sorted(present)),
            "yazmayan_ulkeler": ";".join(missing),
            "kapsama_orani": round(len(present) / len(tracked), 3) if tracked else 0.0,
        })
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GAP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Eksik haber analizi yazıldı: %s (%d küme, izlenen ülke: %d)",
                target, len(rows), len(tracked))
    return target


# ---------------------------------------------------------------- raporlama

def _load_cluster_groups(path: Path | None = None) -> dict[str, list[dict]]:
    target = Path(path) if path else _path(CLUSTERS_CSV)
    if not target.exists():
        raise FileNotFoundError(f"Küme CSV'si bulunamadı: {target}")
    groups: dict[str, list[dict]] = defaultdict(list)
    with target.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups[row["kume_id"]].append(row)
    return dict(groups)


def print_clusters(csv_path: Path | None = None, ilk_n: int = 15,
                   rastgele_kucuk: int = 0, seed: int = 0) -> None:
    """Kümeleri başlık ve kaynaklarıyla yazdırır: en büyük `ilk_n` + isteğe bağlı rastgele küçükler."""
    groups = _load_cluster_groups(csv_path)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    secili = ordered[:max(0, ilk_n)]
    if rastgele_kucuk and len(ordered) > len(secili):
        pool = ordered[len(secili):]
        secili += random.Random(seed).sample(pool, min(rastgele_kucuk, len(pool)))
    print(f"\n{len(groups)} küme (gösterilen: {len(secili)})")
    for cluster_id, members in secili:
        countries = sorted({m["ulke"] for m in members if m["ulke"]})
        print(f"\n[{cluster_id}] {len(members)} haber · {len(countries)} ülke ({', '.join(countries)})")
        for m in members:
            print(f"   - ({m['kaynak']}/{m['ulke'] or '?'}) {m['baslik'][:100]}")


def single_country_cluster_ids(path: Path | None = None) -> set[str]:
    """Tek ülkeli kümelerin kimlikleri.

    config.FRAME_SKIP_SINGLE_COUNTRY bayrağı açıldığında frame comparator bu kümeleri
    LLM karşılaştırmasına göndermemelidir; bayrak şu an varsayılan olarak kapalı ve
    entegrasyon henüz bağlanmadı.
    """
    if not FRAME_SKIP_SINGLE_COUNTRY:
        return set()
    groups = _load_cluster_groups(path)
    return {cluster_id for cluster_id, members in groups.items()
            if all(m.get("tek_ulkeli") == "evet" for m in members)}


def threshold_table(rows: list[dict], thresholds: tuple[float, ...] = THRESHOLD_GRID,
                    model=None, max_days: int = CLUSTER_MAX_DAYS,
                    max_size: int = CLUSTER_MAX_SIZE) -> list[dict]:
    """Eşik taraması: her eşik için küme istatistikleri (embedding bir kez hesaplanır)."""
    if len(rows) < MIN_CLUSTER_SIZE:
        return []
    vectors = embed_titles([row["title"] for row in rows], model=model)
    table = []
    for threshold in thresholds:
        clusters = _cluster_from_vectors(rows, vectors, threshold=threshold,
                                         max_days=max_days, max_size=max_size)
        sizes = [len(cluster) for cluster in clusters]
        multi_country = []
        for cluster in clusters:
            countries = {m["ulke"] for m in cluster if m["ulke"]}
            multi_country.append(len(countries))
        table.append({
            "esik": threshold,
            "kume_sayisi": len(clusters),
            "kumelenen_haber": sum(sizes),
            "kumelenme_orani": round(sum(sizes) / len(rows), 3) if rows else 0.0,
            "ulke_2_plus": sum(1 for c in multi_country if c >= 2),
            "ulke_3_plus": sum(1 for c in multi_country if c >= 3),
            "ortalama_boyut": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        })
    return table


def old_method_comparison(clusters: list[list[dict]], frame_json_path: Path | None = None) -> dict:
    """Eski (aktör/kelime) yönteminin gruplarıyla karşılaştırma — yalnızca dosya okur."""
    path = Path(frame_json_path) if frame_json_path else _path("data/processed/frame_comparison.json")
    result = {"eski_grup": 0, "tek_parca": 0, "kismi": 0, "az_uyeli": 0, "eksik_uyeler": 0,
              "tr_en_eski": 0, "tr_en_yeni": 0}
    cluster_of: dict[tuple[str, str], str] = {}
    lang_of: dict[tuple[str, str], str] = {}
    for index, members in enumerate(clusters, start=1):
        for member in members:
            cluster_of[(member["source"], member["title"])] = f"K{index:03d}"
            lang_of[(member["source"], member["title"])] = member["dil"]
    result["tr_en_yeni"] = sum(
        1 for members in clusters
        if {"tr", "en"} <= {m["dil"] for m in members if m["dil"]}
    )
    if not path.exists():
        logger.warning("Eski yöntem çıktısı yok (%s), karşılaştırma atlandı", path)
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    groups = data.get("gruplar") or []
    result["eski_grup"] = len(groups)
    for group in groups:
        member_ids = {(h.get("kaynak"), h.get("baslik")) for h in group.get("haberler") or []}
        # Kümelemeden çıkarılan başlıklar (spor, hariç kaynaklar) eşleşmez; sayısı raporlanır.
        matched = [m for m in member_ids if m in cluster_of]
        result["eksik_uyeler"] += len(member_ids) - len(matched)
        if len(matched) < MIN_CLUSTER_SIZE:
            result["az_uyeli"] += 1
            continue
        found = {cluster_of[m] for m in matched}
        if len(found) == 1:
            result["tek_parca"] += 1
        else:
            result["kismi"] += 1
        if {"tr", "en"} <= {lang_of[m] for m in matched}:
            result["tr_en_eski"] += 1
    return result


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agora - embedding tabanlı olay kümeleme (yerel)")
    parser.add_argument("--input", default=str(_path(PROCESSED_CSV)), help="etiketlenmiş haber CSV'si")
    parser.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD)
    parser.add_argument("--max-days", type=int, default=CLUSTER_MAX_DAYS)
    parser.add_argument("--max-size", type=int, default=CLUSTER_MAX_SIZE)
    parser.add_argument("--clusters", default=str(_path(CLUSTERS_CSV)))
    parser.add_argument("--gaps", default=str(_path(GAPS_CSV)))
    parser.add_argument("--compare", action="store_true", help="0.5/0.6/0.7 eşik taraması ve eski yöntem karşılaştırması")
    parser.add_argument("--show", action="store_true", help="kümeleri yazdır")
    parser.add_argument("--top", type=int, default=10, help="yazdırılacak en büyük küme sayısı")
    parser.add_argument("--random-small", type=int, default=5, help="ek rastgele küçük küme sayısı")
    args = parser.parse_args(argv)

    rows, skipped = prepare_rows(load_rows(Path(args.input)))
    logger.info("Kümelenecek haber: %d (atlanan: %s)", len(rows), dict(skipped))
    if len(rows) < MIN_CLUSTER_SIZE:
        logger.error("Kümeleme için yeterli haber yok")
        return

    clusters = cluster_rows(rows, threshold=args.threshold, max_days=args.max_days,
                            max_size=args.max_size)
    logger.info("Eşik %.2f: %d küme", args.threshold, len(clusters))
    write_clusters(clusters, Path(args.clusters))
    write_gaps(clusters, Path(args.gaps))

    if args.compare:
        print("\n=== eşik taraması (yerel, API çağrısı yok) ===")
        print(f"{'eşik':>6}{'küme':>8}{'kümelenen':>11}{'oran':>8}{'2+ ülke':>9}{'3+ ülke':>9}{'ort. boyut':>12}")
        for row in threshold_table(rows, model=None, max_days=args.max_days, max_size=args.max_size):
            print(f"{row['esik']:>6.2f}{row['kume_sayisi']:>8}{row['kumelenen_haber']:>11}"
                  f"{row['kumelenme_orani']:>8.2f}{row['ulke_2_plus']:>9}{row['ulke_3_plus']:>9}"
                  f"{row['ortalama_boyut']:>12.2f}")
        karsilastirma = old_method_comparison(clusters)
        print("\n=== eski yöntemle karşılaştırma ===")
        print(f"  eski grup sayısı: {karsilastirma['eski_grup']}")
        print(f"  embedding kümesinde tek parça bulunan: {karsilastirma['tek_parca']}")
        print(f"  birden fazla kümeye dağılan: {karsilastirma['kismi']}")
        print(f"  TR+EN birleştiren küme — eski: {karsilastirma['tr_en_eski']}, yeni: {karsilastirma['tr_en_yeni']}")

    if args.show:
        print_clusters(Path(args.clusters), ilk_n=args.top, rastgele_kucuk=args.random_small)


if __name__ == "__main__":
    main()
