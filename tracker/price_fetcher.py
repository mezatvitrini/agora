"""Agora - yfinance üzerinden kapanış fiyatı çekici.

Fiyatlar günlük kapanış (Close) olarak alınır. Tatil/hafta sonu gibi fiyat olmayan
günlerde, istenen tarihten önceki en yakın kapanış kullanılır.
"""
from __future__ import annotations
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

logger = logging.getLogger("agora.price")

# Projede kabul edilen kısa varlık adları ve yfinance sembolleri.
ASSET_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "XU100": "XU100.IS",
    "XU030": "XU030.IS",
    "BIST100": "XU100.IS",
}
CRYPTO_ASSETS = {"BTC", "ETH", "SOL", "XRP", "AVAX", "DOGE", "ADA", "BNB"}
BENCHMARK_CRYPTO = "BTC-USD"
BENCHMARK_EQUITY = "XU100.IS"

# Tatile denk gelen tarihlerde en fazla kaç gün geriye bakılır.
LOOKBACK_DAYS = 10


def _ensure_ca_bundle():
    """yfinance'in curl'ü, ASCII olmayan kullanıcı yolundaki sertifika dosyasını okuyamıyor.

    certifi paketini geçici bir ASCII yola kopyalayıp curl'e gösteriyoruz.
    """
    if os.environ.get("CURL_CA_BUNDLE") and os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = Path(certifi.where())
    if not bundle.exists() or str(bundle).isascii():
        return
    target = Path(tempfile.gettempdir()) / "agora_ca_bundle.pem"
    try:
        if not target.exists() or target.stat().st_size != bundle.stat().st_size:
            target.write_bytes(bundle.read_bytes())
    except OSError as exc:
        logger.warning("Sertifika paketi kopyalanamadı: %s", exc)
        return
    os.environ.setdefault("CURL_CA_BUNDLE", str(target))
    os.environ.setdefault("SSL_CERT_FILE", str(target))


def to_symbol(asset: str) -> str:
    """Kısa varlık adını yfinance sembolüne çevirir (BTC -> BTC-USD)."""
    key = (asset or "").strip().upper()
    return ASSET_SYMBOLS.get(key, key)


def is_crypto(asset: str) -> bool:
    key = (asset or "").strip().upper()
    return key in CRYPTO_ASSETS or key.endswith("-USD") or key.endswith("USDT")


def benchmark_symbol(asset: str) -> str:
    """Anormal getiri için referans: kriptoda BTC-USD, hisselerde XU100.IS."""
    return BENCHMARK_CRYPTO if is_crypto(asset) else BENCHMARK_EQUITY


def fetch_history(symbol: str, start: date, end: date) -> pd.Series:
    """[start, end] aralığındaki günlük kapanışları tarih indeksli Series olarak döndürür."""
    _ensure_ca_bundle()
    # yfinance, sertifika ayarı yapıldıktan sonra import edilmeli.
    import yfinance as yf

    try:
        frame = yf.Ticker(symbol).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
        )
    except Exception as exc:  # ağ/SSL hataları yfinance içinde farklı tiplerde geliyor
        logger.warning("Fiyat çekilemedi: %s (%s: %s)", symbol, type(exc).__name__, exc)
        return pd.Series(dtype="float64")
    if frame is None or frame.empty or "Close" not in frame:
        logger.warning("Fiyat verisi yok: %s (%s - %s)", symbol, start, end)
        return pd.Series(dtype="float64")

    close = frame["Close"].dropna().astype(float)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index = [ts.date() for ts in close.index]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def latest_date(series: pd.Series):
    """Serideki en yeni fiyat tarihi; seri boşsa None."""
    if series is None or series.empty:
        return None
    return max(series.index)


def close_on_or_before(series: pd.Series, target: date, lookback_days: int = LOOKBACK_DAYS):
    """Hedef tarihteki (yoksa en yakın önceki) kapanışı (fiyat, tarih) olarak döndürür."""
    if series is None or series.empty:
        return None
    eligible = [day for day in series.index if day <= target]
    if not eligible:
        return None
    day = max(eligible)
    if (target - day).days > lookback_days:
        return None
    return float(series.loc[day]), day


def get_close_price(asset: str, on: date, lookback_days: int = LOOKBACK_DAYS):
    """Bir varlığın belirli bir tarihteki kapanış fiyatı (fiyat, gerçek tarih) döner."""
    symbol = to_symbol(asset)
    series = fetch_history(symbol, on - timedelta(days=lookback_days), on)
    return close_on_or_before(series, on, lookback_days)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for name in ("BTC", "ETH", "XU100"):
        result = get_close_price(name, date.today() - timedelta(days=5))
        print(name, to_symbol(name), "->", result)
