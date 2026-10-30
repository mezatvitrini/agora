"""Agora - tahmin kaydı (predictions.csv) yazıcısı.

DeepSeek'ten gelen yön/güven çıkarımını tek bir CSV dosyasında tutar.
Sütunlar sabittir; değerlendirme sonuçları evaluator tarafından doldurulur.
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

COLUMNS = [
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

DIRECTION_LABELS = {
    "olumlu": "olumlu",
    "pozitif": "olumlu",
    "olumsuz": "olumsuz",
    "negatif": "olumsuz",
    "notr": "nötr",
    "nötr": "nötr",
}

DURUM_PENDING = "beklemede"
DURUM_DONE = "tamamlandi"


def load_rows(path: Path | None = None) -> list[dict]:
    """CSV'yi okur; dosya yoksa başlıkla birlikte oluşturup boş liste döner."""
    target = Path(path) if path else PREDICTIONS_CSV
    if not target.exists():
        save_rows([], target)
        logger.info("Tahmin dosyası oluşturuldu: %s", target)
        return []
    with target.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def save_prediction(headline: str, asset: str, direction: str, confidence, tarih=None) -> dict:
    """LLM çıkarımını predictions.csv'ye 'beklemede' durumuyla ekler.

    tarih: geriye dönük kayıt/backfill için isteğe bağlı (YYYY-AA-GG), varsayılan bugün.
    Dönen değer yazılan satırdır.
    """
    title = str(headline or "").strip()
    if not title:
        raise ValueError("Haber başlığı boş olamaz")
    normalized_asset = str(asset or "").strip().upper()
    if not normalized_asset:
        raise ValueError("Varlık boş olamaz")
    prediction_day = _parse_date(tarih) or date.today()

    rows = load_rows()
    row = {
        "tahmin_id": _next_id(rows),
        "tahmin_tarihi": prediction_day.isoformat(),
        "haber_basligi": title,
        "varlik": normalized_asset,
        "ai_tahmini": _normalize_direction(direction),
        "ai_guven_skoru": str(_normalize_confidence(confidence)),
        "gerceklesen_getiri_1g": "",
        "gerceklesen_getiri_7g": "",
        "durum": DURUM_PENDING,
    }
    rows.append(row)
    save_rows(rows)
    logger.info("Tahmin kaydedildi: %s | %s | %s | %s", row["tahmin_id"], row["varlik"],
                row["ai_tahmini"], row["tahmin_tarihi"])
    return row
