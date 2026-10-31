"""Agora - tahmin değerlendirici.

Bekleyen tahminler için:
- Başlangıç fiyatı: tahmin anından SONRAKİ ilk kapanış (look-ahead önleme;
  kripto 7/24 olduğu için tahmin gününün kapanışı, borsa için sonraki işlem günü kapanışı).
- Ham getiri (1g/7g) ve referans varsa anormal getiri (ham - referans getirisi).
- Referans: config.REFERANS_VARLIK (yoksa BIST hisseleri XU100.IS). Referans yoksa
  anormal getiri boş bırakılır, ölçüm ham getiriye dayanır ("olcum_tipi" = ham).
- isabet: varlik_yon ile seçilen getirinin (anormal varsa anormal, yoksa ham) işareti
  uyuşuyorsa True; notr için |getiri| < %1 ise True.
Değerler yüzde puan olarak yazılır (1.25 = %1,25).
"""
from __future__ import annotations
import logging
from datetime import date, timedelta
from pathlib import Path

from tracker.price_fetcher import (
    close_on_or_after,
    close_on_or_before,
    fetch_history,
    get_start_price,
    is_crypto,
    referans_varlik,
    to_symbol,
)
from tracker.prediction_tracker import (
    DURUM_DONE,
    DURUM_PENDING,
    OLCUM_ALANLARI,
    PREDICTIONS_CSV,
    YON_FROM_DUYGU,
    load_rows,
    migrate_rows,
    save_rows,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("agora.evaluator")

HORIZONS = (1, 7)
EXTRA_DAYS = 3
NOTR_ESIK = 1.0  # notr tahminde |getiri| bu yüzdenin altındaysa isabet


def _parse_day(value: str) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        logger.warning("Tahmin tarihi okunamadı: %r", value)
        return None


def _pct(value: float) -> str:
    return f"{value:.2f}"


def _isabet(yon: str, getiri: float | None) -> str:
    if getiri is None:
        return ""
    if yon == "yukari":
        return "True" if getiri > 0 else "False"
    if yon == "asagi":
        return "True" if getiri < 0 else "False"
    return "True" if abs(getiri) < NOTR_ESIK else "False"


def _horizon_price(series, start_day: date, horizon: int):
    hedef = start_day + timedelta(days=horizon)
    return close_on_or_after(series, hedef, lookahead_days=EXTRA_DAYS + 2)


def _referans_degeri(ref_series, start_day: date, horizon: int):
    hedef = start_day + timedelta(days=horizon)
    return close_on_or_after(ref_series, hedef, lookahead_days=EXTRA_DAYS + 2)


def evaluate_predictions(path: Path | None = None, tahmin_ids: list[str] | None = None,
                         kismi: bool = False) -> dict:
    """Bekleyen tahminleri (veya verilen tahmin_id'leri) değerlendirir.

    kismi=True: yalnızca 1g dolmuşsa da yazılır (7g boş kalır, durum 'beklemede' kalır);
    replay akışında 7 günü dolmamış kayıtlar için kullanılır.
    """
    target = Path(path) if path else PREDICTIONS_CSV
    if not target.exists():
        logger.error("Tahmin dosyası bulunamadı: %s", target)
        return {"toplam": 0, "beklemede": 0, "tamamlanan": 0, "bekleyen": 0, "detay": []}

    rows = load_rows(target)
    rows, _ = migrate_rows(rows)
    if tahmin_ids:
        istenen = {str(t).strip() for t in tahmin_ids}
        pending = [row for row in rows if str(row.get("tahmin_id") or "").strip() in istenen]
        logger.info("%d tahmin yeniden değerlendirilecek: %s", len(pending), ", ".join(sorted(istenen)))
    else:
        pending = [row for row in rows if str(row.get("durum") or "").strip() == DURUM_PENDING]
        logger.info("%d tahmin okundu, %d tanesi beklemede", len(rows), len(pending))

    summary = {"toplam": len(rows), "beklemede": len(pending), "tamamlanan": 0,
               "bekleyen": 0, "detay": []}
    if not pending:
        logger.info("Değerlendirilecek beklemede tahmin yok")
        return summary

    for row in pending:
        tahmin_id = row.get("tahmin_id", "?")
        day = _parse_day(row.get("tahmin_tarihi", ""))
        asset = str(row.get("varlik") or "").strip()
        if day is None or not asset:
            summary["bekleyen"] += 1
            continue
        symbol = to_symbol(asset)
        referans = referans_varlik(asset)
        surekli = is_crypto(asset) or symbol.endswith("=X") or symbol.endswith("=F") or symbol.startswith("^")

        baslangic = get_start_price(asset, day)
        if baslangic is None or baslangic[0] is None:
            logger.warning("%s: başlangıç fiyatı yok, beklemede", tahmin_id)
            summary["bekleyen"] += 1
            continue
        baslangic_fiyat, baslangic_gun = baslangic

        pencere_bas = baslangic_gun
        pencere_son = min(baslangic_gun + timedelta(days=max(HORIZONS) + EXTRA_DAYS), date.today())
        asset_series = fetch_history(symbol, pencere_bas, pencere_son)
        if asset_series.empty:
            summary["bekleyen"] += 1
            continue

        ref_series = None
        ref_baslangic = None
        if referans:
            ref_series = fetch_history(referans, pencere_bas, pencere_son)
            ref_start = close_on_or_before(ref_series, baslangic_gun)
            if ref_start is None:
                ref_start = close_on_or_after(ref_series, baslangic_gun)
            ref_baslangic = ref_start[0] if ref_start else None
            if ref_baslangic is None:
                logger.info("%s: referans (%s) başlangıç fiyatı yok; ölçüm ham getiriye düşecek",
                            tahmin_id, referans)
                ref_series = None

        ham: dict[int, float] = {}
        anormal: dict[int, float] = {}
        eksik = False
        for horizon in HORIZONS:
            fiyat = _horizon_price(asset_series, baslangic_gun, horizon)
            if fiyat is None:
                eksik = True
                break
            ham[horizon] = (fiyat[0] / baslangic_fiyat - 1) * 100
            if ref_series is not None and ref_baslangic:
                ref_fiyat = _referans_degeri(ref_series, baslangic_gun, horizon)
                if ref_fiyat is None:
                    eksik = True
                    break
                ref_getiri = (ref_fiyat[0] / ref_baslangic - 1) * 100
                anormal[horizon] = ham[horizon] - ref_getiri
        if eksik and not kismi:
            logger.info("%s: horizon fiyatları henüz yok, beklemede bırakıldı", tahmin_id)
            summary["bekleyen"] += 1
            continue
        if not ham:
            summary["bekleyen"] += 1
            continue

        olcum_tipi = "anormal" if anormal else "ham"
        yon = (row.get("varlik_yon") or "").strip() or YON_FROM_DUYGU.get(
            (row.get("haber_ton") or row.get("ai_tahmini") or "").strip(), "notr")

        tamam = len(ham) == len(HORIZONS)
        deger = {
            "baslangic_fiyat": _pct(baslangic_fiyat) if baslangic_fiyat is not None else "",
            "baslangic_zamani": baslangic_gun.isoformat(),
            "getiri_ham_1g": _pct(ham[1]) if 1 in ham else "",
            "getiri_ham_7g": _pct(ham[7]) if 7 in ham else "",
            "anormal_getiri_1g": _pct(anormal[1]) if 1 in anormal else "",
            "anormal_getiri_7g": _pct(anormal[7]) if 7 in anormal else "",
            "olcum_tipi": olcum_tipi,
            "isabet_1g": (_isabet(yon, anormal.get(1) if olcum_tipi == "anormal" else ham.get(1))
                          if 1 in ham else ""),
            "isabet_7g": (_isabet(yon, anormal.get(7) if olcum_tipi == "anormal" else ham.get(7))
                          if 7 in ham else ""),
            "durum": DURUM_DONE if tamam else DURUM_PENDING,
        }
        izinsiz = set(deger) - OLCUM_ALANLARI
        if izinsiz:  # kayıt değiştirilemez ilkesinin koruması
            raise ValueError(f"Ölçüm dışı alan yazılmaya çalışıldı: {sorted(izinsiz)}")
        row.update({k: str(v) for k, v in deger.items()})
        row["referans_varlik"] = referans or ""

        summary["tamamlanan"] += 1
        summary["detay"].append({
            "tahmin_id": tahmin_id,
            "varlik": asset,
            "referans": referans,
            "olcum_tipi": olcum_tipi,
            "baslangic_fiyat": deger["baslangic_fiyat"],
            "baslangic_zamani": deger["baslangic_zamani"],
            "getiri_ham_1g": deger["getiri_ham_1g"],
            "getiri_ham_7g": deger["getiri_ham_7g"],
            "anormal_getiri_1g": deger["anormal_getiri_1g"],
            "anormal_getiri_7g": deger["anormal_getiri_7g"],
            "isabet_1g": deger["isabet_1g"],
            "isabet_7g": deger["isabet_7g"],
        })
        logger.info("%s: %s | %s | ham 1g=%s 7g=%s | anormal 1g=%s 7g=%s | isabet 1g=%s 7g=%s",
                    tahmin_id, asset, olcum_tipi, deger["getiri_ham_1g"], deger["getiri_ham_7g"],
                    deger["anormal_getiri_1g"] or "—", deger["anormal_getiri_7g"] or "—",
                    deger["isabet_1g"], deger["isabet_7g"])

    save_rows(rows, target)
    logger.info("Değerlendirme bitti: %d tamamlandı, %d beklemede kaldı",
                summary["tamamlanan"], summary["bekleyen"])
    return summary


# ---------------------------------------------------------------- istatistikler

def _secili_getiri(row: dict, horizon: int) -> float | None:
    kolon = ("anormal_getiri" if (row.get("olcum_tipi") or "").strip() == "anormal"
             else "getiri_ham") + f"_{horizon}g"
    try:
        return float(str(row.get(kolon) or "").replace(",", "."))
    except ValueError:
        return None


def _oran(rows: list[dict], horizon: int) -> float | None:
    degerler = []
    for row in rows:
        isabet = str(row.get(f"isabet_{horizon}g") or "").strip().lower()
        if isabet in ("true", "false"):
            degerler.append(1.0 if isabet == "true" else 0.0)
    return round(sum(degerler) / len(degerler), 3) if degerler else None


def _slice(rows: list[dict], min_dilim: int) -> dict:
    kullanilabilir = [r for r in rows
                      if str(r.get("isabet_1g") or "").strip().lower() in ("true", "false")]
    n = len(kullanilabilir)
    sonuc = {"n": n, "yetersiz": n < min_dilim}
    for horizon in HORIZONS:
        oran = _oran(kullanilabilir, horizon)
        sonuc[f"isabet_{horizon}g"] = None if sonuc["yetersiz"] else oran
        sonuc[f"oran_{horizon}g"] = oran  # yetersiz dilimlerde de bilgi amaçlı
    return sonuc


def _guven_araligi(deger) -> str:
    try:
        skor = int(str(deger).strip())
    except ValueError:
        return "bilinmiyor"
    if skor <= 3:
        return "1-3"
    if skor <= 6:
        return "4-6"
    return "7-10"


def get_summary_stats(path: Path | None = None, min_dilim: int = 30) -> dict:
    """Tamamlanan tahminler üzerinden isabet istatistikleri (dilim başına en az `min_dilim`)."""
    target = Path(path) if path else PREDICTIONS_CSV
    rows = load_rows(target)
    rows, _ = migrate_rows(rows)
    tamamlanan = [row for row in rows if str(row.get("durum") or "").strip() == DURUM_DONE]

    def dilimle(anahtar) -> dict:
        gruplar: dict[str, list[dict]] = {}
        for row in tamamlanan:
            gruplar.setdefault(anahtar(row) or "—", []).append(row)
        return {ad: _slice(grup, min_dilim) for ad, grup in sorted(gruplar.items())}

    veri_tipleri = sorted({(row.get("veri_tipi") or "canli").strip() for row in tamamlanan})
    baseline = {}
    for tip in veri_tipleri:
        grup = [r for r in tamamlanan if (r.get("veri_tipi") or "canli").strip() == tip]
        degerler = [g for g in (_secili_getiri(r, 1) for r in grup) if g is not None]
        yukari = [g for g in degerler if g > 0]
        baseline[tip] = (round(len(yukari) / len(degerler), 3) if degerler else None, len(degerler))

    return {
        "toplam": len(rows),
        "tamamlanan": len(tamamlanan),
        "beklemede": len(rows) - len(tamamlanan),
        "min_dilim": min_dilim,
        "genel": _slice(tamamlanan, min_dilim),
        "veri_tipi": {tip: _slice([r for r in tamamlanan
                                   if (r.get("veri_tipi") or "canli").strip() == tip], min_dilim)
                      for tip in veri_tipleri},
        "model": dilimle(lambda r: r.get("model_adi")),
        "varlik": dilimle(lambda r: r.get("varlik")),
        "kategori": dilimle(lambda r: r.get("kategori")),
        "guven": dilimle(lambda r: _guven_araligi(r.get("ai_guven_skoru"))),
        "baseline_isabet": baseline,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agora - tahmin değerlendirici")
    parser.add_argument("--yeniden", nargs="*", default=None,
                        help="Verilen tahmin_id'leri durumdan bağımsız yeniden değerlendir")
    args = parser.parse_args()
    evaluate_predictions(tahmin_ids=args.yeniden)
    print(get_summary_stats())
