"""Agora - tahmin değerlendirici.

'beklemede' olan tahminleri okur, tahmin tarihinden 1 ve 7 gün sonraki kapanışı
price_fetcher ile çeker ve anormal getiriyi (varlık getirisi - referans endeks
getirisi) hesaplayarak satırı 'tamamlandi' yapar.

Anormal getiri: kripto varlıklarda referans BTC-USD, hisselerde XU100.IS.
Değerler yüzde puan olarak yazılır (1.25 = %1,25).
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from tracker.price_fetcher import (
    LOOKBACK_DAYS,
    benchmark_symbol,
    close_on_or_before,
    fetch_history,
    latest_date,
    to_symbol,
)
from tracker.prediction_tracker import (
    DURUM_DONE,
    DURUM_PENDING,
    PREDICTIONS_CSV,
    load_rows,
    save_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.evaluator")

HORIZONS = (1, 7)
RETURN_COLUMNS = {1: "gerceklesen_getiri_1g", 7: "gerceklesen_getiri_7g"}
EXTRA_DAYS = 3  # son horizon için fiyatın yayınlanmasını bekleme payı


def _parse_day(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        logger.warning("Tahmin tarihi okunamadı: %r", value)
        return None


def _anomaly(asset_series: pd.Series, bench_series: pd.Series, base_day: date, target_day: date):
    """(varlık getirisi - referans getirisi) yüzde puan; veri eksikse None."""
    asset_base = close_on_or_before(asset_series, base_day)
    bench_base = close_on_or_before(bench_series, base_day)
    asset_now = close_on_or_before(asset_series, target_day)
    bench_now = close_on_or_before(bench_series, target_day)
    if not all((asset_base, bench_base, asset_now, bench_now)):
        return None
    if not all(price[0] > 0 for price in (asset_base, bench_base)):
        return None
    asset_return = (asset_now[0] / asset_base[0] - 1) * 100
    bench_return = (bench_now[0] / bench_base[0] - 1) * 100
    value = round(asset_return - bench_return, 2)
    return 0.0 if abs(value) < 0.005 else value


def _write_result(row: dict, values: dict[int, float]) -> None:
    row["durum"] = DURUM_DONE
    for horizon, column in RETURN_COLUMNS.items():
        row[column] = f"{values[horizon]:.2f}"


def evaluate_predictions(path: Path | None = None) -> dict:
    """Bekleyen tahminleri değerlendirir; özet sözlüğü döndürür."""
    target = Path(path) if path else PREDICTIONS_CSV
    if not target.exists():
        logger.error("Tahmin dosyası bulunamadı: %s", target)
        return {"toplam": 0, "beklemede": 0, "tamamlanan": 0, "bekleyen": 0, "detay": []}

    rows = load_rows(target)
    pending = [row for row in rows if str(row.get("durum") or "").strip() == DURUM_PENDING]
    logger.info("%d tahmin okundu, %d tanesi beklemede", len(rows), len(pending))

    summary = {"toplam": len(rows), "beklemede": len(pending), "tamamlanan": 0,
               "bekleyen": 0, "detay": []}
    if not pending:
        logger.info("Değerlendirilecek beklemede tahmin yok")
        return summary

    prediction_days = [_parse_day(row.get("tahmin_tarihi", "")) for row in pending]
    valid_days = [day for day in prediction_days if day]
    if not valid_days:
        logger.warning("Geçerli tahmin tarihi bulunamadı")
        return summary

    start = min(valid_days) - timedelta(days=LOOKBACK_DAYS)
    # Gelecekteki tarihleri istemek yfinance'ten boş yanıt döndürür; bugünle sınırlıyoruz.
    end = min(max(valid_days) + timedelta(days=max(HORIZONS) + EXTRA_DAYS), date.today())
    cache: dict[str, pd.Series] = {}

    def series(symbol: str) -> pd.Series:
        if symbol not in cache:
            cache[symbol] = fetch_history(symbol, start, end)
        return cache[symbol]

    for row, base_day in zip(pending, prediction_days):
        prediction_id = row.get("tahmin_id", "?")
        if base_day is None:
            summary["bekleyen"] += 1
            continue
        asset = str(row.get("varlik") or "").strip()
        symbol = to_symbol(asset)
        if not symbol:
            logger.warning("%s: varlık boş, atlandı", prediction_id)
            summary["bekleyen"] += 1
            continue
        bench = benchmark_symbol(asset)
        if bench == symbol:
            logger.info("%s: %s kendi referansı, anormal getiri 0 olacak", prediction_id, symbol)

        asset_series = series(symbol)
        bench_series = series(bench)
        # Fiyat serisi hedef güne ulaşmadıysa (gelecek tarih) değerlendirme yapılmaz.
        asset_latest = latest_date(asset_series)
        bench_latest = latest_date(bench_series)

        values: dict[int, float] = {}
        incomplete = False
        for horizon in HORIZONS:
            target_day = base_day + timedelta(days=horizon)
            if asset_latest is None or asset_latest < target_day:
                incomplete = True
                break
            if bench_latest is None or bench_latest < target_day:
                incomplete = True
                break
            value = _anomaly(asset_series, bench_series, base_day, target_day)
            if value is None:
                incomplete = True
                break
            values[horizon] = value
        if incomplete:
            logger.info("%s: %d günlük fiyat henüz yok, beklemede bırakıldı",
                        prediction_id, HORIZONS[len(values)])
            summary["bekleyen"] += 1
            continue

        _write_result(row, values)
        summary["tamamlanan"] += 1
        summary["detay"].append({
            "tahmin_id": prediction_id,
            "varlik": asset,
            "referans": bench,
            "tahmin_tarihi": base_day.isoformat(),
            "anormal_getiri_1g": values[1],
            "anormal_getiri_7g": values[7],
        })
        logger.info("%s: %s 1g=%.2f 7g=%.2f (referans %s)", prediction_id, asset,
                    values[1], values[7], bench)

    save_rows(rows, target)
    logger.info("Değerlendirme bitti: %d tamamlandı, %d beklemede kaldı",
                summary["tamamlanan"], summary["bekleyen"])
    return summary


if __name__ == "__main__":
    evaluate_predictions()
