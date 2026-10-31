"""Agora - tahmin kaydı (predictions.csv) yazıcısı.

Kurallar:
- Kayıt değiştirilemez: save_prediction yalnızca YENİ satır ekler.
- Sonradan yalnızca ölçüm sütunları doldurulabilir (baslangic_fiyat, getiriler, isabet,
  durum); bunun dışındaki alanlar update_measurements tarafından reddedilir.
- alinti_kanit zorunludur: boşsa kayıt reddedilir (kanıtsız tahmin kaydedilmez).
"""
from __future__ import annotations
import csv
import logging
import os
import time
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.tracker")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions.csv"

# Eski (v1) sütunlar korunuyor; yeni sütunlar sonda.
LEGACY_COLUMNS = [
    "tahmin_id",
    "tahmin_tarihi",
    "haber_basligi",
    "varlik",
    "ai_tahmini",
    "ai_guven_skoru",
    "gerceklesen_getiri_1g",
    "gerceklesen_getiri_7g",
    "durum",
]
NEW_COLUMNS = [
    "kume_id",
    "model_adi",
    "prompt_versiyonu",
    "kategori",
    "referans_varlik",
    "haber_ton",
    "varlik_yon",
    "alinti_kanit",
    "gerekce",
    "getiri_ham_1g",
    "getiri_ham_7g",
    "anormal_getiri_1g",
    "anormal_getiri_7g",
    "olcum_tipi",
    "isabet_1g",
    "isabet_7g",
    "veri_tipi",
    "donem_tipi",
    "baslangic_fiyat",
    "baslangic_zamani",
    "kopya",
    "eski_icerik",
]
COLUMNS = LEGACY_COLUMNS + NEW_COLUMNS

# Sonradan doldurulabilen tek alanlar (kayıt değiştirilemez ilkesi)
OLCUM_ALANLARI = {
    "baslangic_fiyat", "baslangic_zamani",
    "getiri_ham_1g", "getiri_ham_7g",
    "anormal_getiri_1g", "anormal_getiri_7g",
    "olcum_tipi", "isabet_1g", "isabet_7g", "durum",
    "gerceklesen_getiri_1g", "gerceklesen_getiri_7g",
}

DIRECTION_LABELS = {
    "olumlu": "olumlu",
    "pozitif": "olumlu",
    "olumsuz": "olumsuz",
    "negatif": "olumsuz",
    "notr": "nötr",
    "nötr": "nötr",
}

YON_LABELS = {
    "yukari": "yukari", "yukarı": "yukari", "up": "yukari", "artis": "yukari", "artış": "yukari",
    "asagi": "asagi", "aşağı": "asagi", "down": "asagi", "dusus": "asagi", "düşüş": "asagi",
    "notr": "notr", "nötr": "notr", "flat": "notr", "yatay": "notr",
}
YON_FROM_DUYGU = {"olumlu": "yukari", "olumsuz": "asagi", "nötr": "notr"}

DURUM_PENDING = "beklemede"
DURUM_DONE = "tamamlandi"
VERI_TIPI_VALUES = {"canli", "geri_tarama"}


def load_rows(path: Path | None = None) -> list[dict]:
    """CSV'yi okur; dosya yoksa başlıkla birlikte oluşturup boş liste döner."""
    target = Path(path) if path else PREDICTIONS_CSV
    if not target.exists():
        save_rows([], target)
        logger.info("Tahmin dosyası oluşturuldu: %s", target)
        return []
    with target.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def migrate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Eksik sütunları boş ekler; eski satırların verisini korur."""
    degisen = 0
    for row in rows:
        eksik = [column for column in COLUMNS if column not in row]
        if eksik:
            degisen += 1
        for column in COLUMNS:
            row.setdefault(column, "")
        if not (row.get("veri_tipi") or "").strip():
            row["veri_tipi"] = "canli"
        if not (row.get("model_adi") or "").strip() and row.get("ai_tahmini"):
            row["model_adi"] = ""
    return rows, degisen


def migrate_file(path: Path | None = None) -> int:
    target = Path(path) if path else PREDICTIONS_CSV
    rows = load_rows(target)
    if not rows:
        return 0
    rows, degisen = migrate_rows(rows)
    if degisen:
        save_rows(rows, target)
    return degisen


def save_rows(rows: list[dict], path: Path | None = None) -> None:
    target = Path(path) if path else PREDICTIONS_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    # Windows'ta virüs tarayıcı/indeksleyici dosyayı kısa süre kilitleyebiliyor.
    for attempt in range(5):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def _normalize_direction(direction: str) -> str:
    key = str(direction or "").strip().lower()
    if key not in DIRECTION_LABELS:
        raise ValueError(f"Geçersiz yön: {direction!r} (olumlu/olumsuz/nötr olmalı)")
    return DIRECTION_LABELS[key]


def _normalize_varlik_yon(value: str | None, fallback: str) -> str:
    if value is None or not str(value).strip():
        return YON_FROM_DUYGU.get(fallback, "notr")
    key = str(value).strip().lower()
    if key not in YON_LABELS:
        raise ValueError(f"Geçersiz varlik_yon: {value!r} (yukari/asagi/notr olmalı)")
    return YON_LABELS[key]


def _normalize_confidence(confidence) -> int:
    try:
        value = int(str(confidence).strip())
    except (TypeError, ValueError):
        raise ValueError(f"Geçersiz güven skoru: {confidence!r} (1-10 arası tam sayı olmalı)")
    if not 1 <= value <= 10:
        raise ValueError(f"Güven skoru 1-10 arasında olmalı: {value}")
    return value


def _next_id(rows: list[dict]) -> str:
    numbers = []
    for row in rows:
        raw = str(row.get("tahmin_id") or "").strip().lstrip("Tt")
        if raw.isdigit():
            numbers.append(int(raw))
    return f"T{max(numbers, default=0) + 1:04d}"


def _parse_date(value) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise ValueError(f"Geçersiz tarih: {value!r} (YYYY-AA-GG biçiminde olmalı)")


def save_prediction(headline: str, asset: str, direction: str, confidence, tarih=None, *,
                    alinti_kanit: str | None = None, kume_id: str = "", model_adi: str = "",
                    prompt_versiyonu: str = "", kategori: str = "", haber_ton: str | None = None,
                    varlik_yon: str | None = None, gerekce: str = "",
                    veri_tipi: str = "canli", donem_tipi: str = "",
                    referans_varlik: str = "") -> dict:
    """Yeni tahmin satırı ekler (daima 'beklemede').

    alinti_kanit zorunludur: boş/None ise ValueError fırlatılır.
    haber_ton verilmezse `direction`, varlik_yon verilmezse haber tonundan türetilir.
    """
    title = str(headline or "").strip()
    if not title:
        raise ValueError("Haber başlığı boş olamaz")
    normalized_asset = str(asset or "").strip().upper()
    if not normalized_asset:
        raise ValueError("Varlık boş olamaz")
    kanit = str(alinti_kanit or "").strip()
    if not kanit:
        raise ValueError("alinti_kanit zorunludur: kanıtsız tahmin kaydedilemez")
    tip = str(veri_tipi or "canli").strip().lower()
    if tip not in VERI_TIPI_VALUES:
        raise ValueError(f"Geçersiz veri_tipi: {veri_tipi!r} (canli/geri_tarama olmalı)")
    donem = str(donem_tipi or "").strip().lower()
    if donem and donem not in ("kesif", "dogrulama"):
        raise ValueError(f"Geçersiz donem_tipi: {donem_tipi!r} (kesif/dogrulama olmalı)")

    duygu = _normalize_direction(direction)
    ton = _normalize_direction(haber_ton) if (haber_ton and str(haber_ton).strip()) else duygu
    prediction_day = _parse_date(tarih) or date.today()

    rows = load_rows()
    rows, _ = migrate_rows(rows)
    seri_no = _next_id(rows)
    row = {column: "" for column in COLUMNS}
    row.update({
        "tahmin_id": seri_no,
        "tahmin_tarihi": prediction_day.isoformat(),
        "haber_basligi": title,
        "varlik": normalized_asset,
        "ai_tahmini": duygu,
        "ai_guven_skoru": str(_normalize_confidence(confidence)),
        "durum": DURUM_PENDING,
        "kume_id": str(kume_id or "").strip(),
        "model_adi": str(model_adi or "").strip(),
        "prompt_versiyonu": str(prompt_versiyonu or "").strip(),
        "kategori": str(kategori or "").strip(),
        "referans_varlik": str(referans_varlik or "").strip(),
        "haber_ton": ton,
        "varlik_yon": _normalize_varlik_yon(varlik_yon, ton),
        "alinti_kanit": kanit,
        "gerekce": str(gerekce or "").strip(),
        "veri_tipi": tip,
        "donem_tipi": donem,
        "kopya": "",
        "eski_icerik": "",
    })
    rows.append(row)
    save_rows(rows)
    logger.info("Tahmin kaydedildi: %s | %s | %s | %s", row["tahmin_id"], row["varlik"],
                row["varlik_yon"], row["tahmin_tarihi"])
    return row


def mark_copies(path: Path | None = None) -> list[dict]:
    """Aynı varlık + aynı varlik_yon ve 24 saat içindeki kayıtları 'kopya' olarak işaretler.

    Kural: 24 saatlik pencerede güveni yüksek olan (eşitlikte daha eski olan) kalır,
    diğerleri kopya=True olur. Kayıt silinmez; yalnızca bayrak yazılır.
    2. tur çalıştırma (idempotent): yeni kopya yoksa değişiklik yapmaz.
    """
    target = Path(path) if path else PREDICTIONS_CSV
    rows = load_rows(target)
    rows, _ = migrate_rows(rows)

    def _guven(row: dict) -> float:
        try:
            return float(str(row.get("ai_guven_skoru") or 0).replace(",", "."))
        except ValueError:
            return 0.0

    def _tarih(row: dict):
        text = str(row.get("tahmin_tarihi") or "").strip()
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None

    gruplar: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if str(row.get("kopya") or "").strip().lower() in ("true", "evet", "1"):
            continue  # zaten işaretli
        anahtar = (str(row.get("varlik") or "").strip(), str(row.get("varlik_yon") or "").strip())
        gruplar.setdefault(anahtar, []).append(row)

    isaretlenen: list[dict] = []
    for (varlik, yon), grup in gruplar.items():
        if len(grup) < 2 or not yon:
            continue
        sirali = sorted((r for r in grup if _tarih(r)), key=lambda r: (_tarih(r), -_guven(r)))
        korunan: dict | None = None
        for row in sirali:
            gun = _tarih(row)
            if korunan is None:
                korunan = row
                continue
            if abs((gun - _tarih(korunan)).days) < 1:
                # Aynı gün/24 saat: güveni yüksek olan kalır
                if _guven(row) > _guven(korunan):
                    korunan["kopya"] = "evet"
                    isaretlenen.append({"tahmin_id": korunan["tahmin_id"], "varlik": varlik,
                                        "yon": yon, "tarih": korunan["tahmin_tarihi"]})
                    korunan = row
                else:
                    row["kopya"] = "evet"
                    isaretlenen.append({"tahmin_id": row["tahmin_id"], "varlik": varlik,
                                        "yon": yon, "tarih": row["tahmin_tarihi"]})
            else:
                korunan = row

    if isaretlenen:
        save_rows(rows, target)
        logger.info("Kopya olarak işaretlenen kayıt: %d", len(isaretlenen))
    return isaretlenen


FLAG_ALANLARI = {"kopya", "eski_icerik"}


def set_flag(tahmin_ids: list[str], flag: str = "eski_icerik", value: str = "evet",
             path: Path | None = None) -> int:
    """Kayıtları silmeden bayrak yazar (kopya / eski_icerik)."""
    if flag not in FLAG_ALANLARI:
        raise ValueError(f"Geçersiz bayrak: {flag} (izinli: {sorted(FLAG_ALANLARI)})")
    rows = load_rows(path)
    rows, _ = migrate_rows(rows)
    hedef = {str(t).strip() for t in tahmin_ids}
    sayi = 0
    for row in rows:
        if row.get("tahmin_id") in hedef and str(row.get(flag) or "") != value:
            row[flag] = value
            sayi += 1
    if sayi:
        save_rows(rows, path)
    return sayi


def update_measurements(tahmin_id: str, values: dict, path: Path | None = None) -> dict:
    """Yalnızca ölçüm sütunlarını günceller; kaydın diğer alanları değiştirilemez."""
    izinsiz = set(values) - OLCUM_ALANLARI
    if izinsiz:
        raise ValueError(f"Bu alanlar güncellenemez (kayıt değiştirilemez): {sorted(izinsiz)}")
    rows = load_rows(path)
    rows, _ = migrate_rows(rows)
    for row in rows:
        if row.get("tahmin_id") == tahmin_id:
            for column, value in values.items():
                row[column] = "" if value is None else str(value)
            save_rows(rows, path)
            return row
    raise KeyError(f"Tahmin bulunamadı: {tahmin_id}")
