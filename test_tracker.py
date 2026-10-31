"""Tracker testleri (yerel; fiyat çekimi için ağ gerekir).

Çalıştırma: python test_tracker.py
Kapsam: şema migrasyonu, alinti_kanit zorunluluğu, kayıt değiştirilemezliği,
referans eşlemesi (BTC ham / ETH -> BTC), isabet hesapları, değerlendirme.
"""
import csv
import tempfile
from datetime import date
from pathlib import Path

from tracker import evaluator as ev
from tracker import prediction_tracker as pt
from tracker.price_fetcher import referans_varlik

GECMIS = "2026-10-15"


def _eski_semali_csv(path: Path) -> None:
    alanlar = ["tahmin_id", "tahmin_tarihi", "haber_basligi", "varlik", "ai_tahmini",
               "ai_guven_skoru", "gerceklesen_getiri_1g", "gerceklesen_getiri_7g", "durum"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=alanlar)
        writer.writeheader()
        writer.writerow({"tahmin_id": "T0001", "tahmin_tarihi": GECMIS,
                         "haber_basligi": "Bitcoin yükselişini sürdürüyor", "varlik": "BTC",
                         "ai_tahmini": "olumlu", "ai_guven_skoru": "7",
                         "gerceklesen_getiri_1g": "0.00", "gerceklesen_getiri_7g": "0.00",
                         "durum": "tamamlandi"})
        writer.writerow({"tahmin_id": "T0002", "tahmin_tarihi": GECMIS,
                         "haber_basligi": "Ethereum ETF girişleri hızlandı", "varlik": "ETH",
                         "ai_tahmini": "olumsuz", "ai_guven_skoru": "6",
                         "gerceklesen_getiri_1g": "-0.31", "gerceklesen_getiri_7g": "-1.47",
                         "durum": "tamamlandi"})


def test_migrasyon(tmp: Path) -> None:
    target = tmp / "predictions.csv"
    _eski_semali_csv(target)
    once = list(csv.DictReader(target.open(encoding="utf-8")))
    degisen = pt.migrate_file(target)
    sonra = pt.load_rows(target)
    assert len(sonra) == 2, "migrasyon satır sayısını değiştirdi"
    assert degisen == 2
    for eski, yeni in zip(once, sonra):
        for alan in pt.LEGACY_COLUMNS:
            if alan == "durum":
                continue
            assert eski[alan] == yeni[alan], f"eski alan bozuldu: {alan}"
    assert all(alan in sonra[0] for alan in pt.COLUMNS), "yeni sütunlar eklenmedi"
    assert sonra[0]["veri_tipi"] == "canli"
    print("[OK] migrasyon: eski 2 satır korundu, yeni sütunlar eklendi, veri_tipi=canli")


def test_alinti_kanit_zorunlu(tmp: Path) -> None:
    pt.PREDICTIONS_CSV = tmp / "kanit.csv"
    try:
        pt.save_prediction("Başlık", "BTC", "olumlu", 5, alinti_kanit="")
        raise AssertionError("alinti_kanit boşken kayıt kabul edildi")
    except ValueError as exc:
        assert "alinti_kanit" in str(exc)
    try:
        pt.save_prediction("Başlık", "BTC", "olumlu", 5)
        raise AssertionError("alinti_kanit verilmeden kayıt kabul edildi")
    except ValueError:
        pass
    print("[OK] alinti_kanit zorunlu: boş/eksik kayıt reddediliyor")


def test_degistirilemezlik(tmp: Path) -> None:
    pt.PREDICTIONS_CSV = tmp / "degismez.csv"
    row = pt.save_prediction("Başlık", "BTC", "olumlu", 5, tarih=GECMIS,
                             alinti_kanit="Bitcoin yükselişini sürdürüyor",
                             gerekce="örnek gerekçe")
    try:
        pt.update_measurements(row["tahmin_id"], {"haber_basligi": "değiştir"})
        raise AssertionError("ölçüm dışı alan güncellenebildi")
    except ValueError:
        pass
    guncel = pt.update_measurements(row["tahmin_id"], {"isabet_1g": "True", "durum": "tamamlandi"})
    assert guncel["isabet_1g"] == "True" and guncel["haber_basligi"] == "Başlık"
    print("[OK] kayıt değiştirilemez: yalnızca ölçüm sütunları güncellenebiliyor")


def test_referans() -> None:
    beklenen = {
        "BTC": None, "BTC-USD": None, "ETH": "BTC-USD", "ETH-USD": "BTC-USD",
        "XU100": None, "XU100.IS": None, "THYAO.IS": "XU100.IS",
        "TRY=X": None, "CL=F": None, "GC=F": None, "NG=F": None, "^GSPC": None,
    }
    for varlik, referans in beklenen.items():
        elde = referans_varlik(varlik)
        assert elde == referans, f"{varlik}: beklenen {referans}, bulunan {elde}"
    print("[OK] referans eşlemesi: BTC/XU100/TRY=/CL=F/GC=F/NG=F/^GSPC -> None, ETH -> BTC-USD, THYAO.IS -> XU100.IS")


def test_isabet() -> None:
    durumlar = [
        ("yukari", 2.0, "True"), ("yukari", -1.0, "False"),
        ("asagi", -0.4, "True"), ("asagi", 1.2, "False"),
        ("notr", 0.5, "True"), ("notr", 2.0, "False"),
        ("yukari", 0.0, "False"),
    ]
    for yon, getiri, beklenen in durumlar:
        elde = ev._isabet(yon, getiri)
        assert elde == beklenen, f"isabet({yon}, {getiri}) = {elde}, beklenen {beklenen}"
    assert ev._isabet("yukari", None) == ""
    print("[OK] isabet hesabı: yukari/asagi işaret, notr için |getiri| < %1")


def test_10_tahmin_ve_degerlendirme(tmp: Path) -> None:
    target = tmp / "on_tahmin.csv"
    pt.PREDICTIONS_CSV = target
    _eski_semali_csv(target)  # önce eski 2 satır (migrasyonla birlikte)
    varliklar = ["BTC", "ETH"] * 5
    for index, varlik in enumerate(varliklar, start=1):
        pt.save_prediction(
            f"Sahte haber {index}", varlik, "olumlu" if index % 2 else "olumsuz",
            5 + (index % 5), tarih=GECMIS,
            alinti_kanit=f"Sahte haber {index}",
            kume_id=f"K{index:03d}", model_adi="deepseek-chat", prompt_versiyonu="v2",
            kategori="ekonomi", gerekce="test",
        )
    rows = pt.load_rows(target)
    assert len(rows) == 12, f"beklenen 12 satır (2 eski + 10 yeni), bulunan {len(rows)}"
    assert rows[0]["tahmin_id"] == "T0001" and rows[0]["gerceklesen_getiri_1g"] == "0.00"
    yeni = [r for r in rows if r["tahmin_id"] not in ("T0001", "T0002")]
    assert all(r["alinti_kanit"] for r in yeni)
    print(f"[OK] 10 sahte tahmin eklendi (toplam {len(rows)} satır), eski 2 satır bozulmadı")

    # BTC ve ETH için birer satırı gerçekten değerlendir (ağ gerekir)
    secilen = [yeni[0]["tahmin_id"], yeni[1]["tahmin_id"]]
    ozet = ev.evaluate_predictions(path=target, tahmin_ids=secilen)
    assert ozet["tamamlanan"] == 2, f"2 satır değerlendirilmeliydi: {ozet}"
    sonuc = {r["tahmin_id"]: r for r in pt.load_rows(target)}
    btc, eth = sonuc[secilen[0]], sonuc[secilen[1]]
    assert btc["varlik"] == "BTC" and btc["olcum_tipi"] == "ham", f"BTC ölçüm tipi: {btc['olcum_tipi']}"
    assert btc["anormal_getiri_1g"] == "", "BTC için anormal getiri boş olmalı"
    assert eth["varlik"] == "ETH" and eth["olcum_tipi"] == "anormal", f"ETH ölçüm tipi: {eth['olcum_tipi']}"
    assert eth["referans_varlik"] == "BTC-USD", f"ETH referansı: {eth['referans_varlik']}"
    assert eth["anormal_getiri_1g"] != "" and eth["isabet_1g"] in ("True", "False")
    assert btc["baslangic_fiyat"] and btc["baslangic_zamani"], "başlangıç fiyatı/zamanı yazılmalı"
    print(f"[OK] BTC olcum_tipi=ham (anormal boş), ETH olcum_tipi=anormal, referans=BTC-USD")
    print(f"     BTC ham 1g={btc['getiri_ham_1g']} | ETH ham 1g={eth['getiri_ham_1g']} "
          f"anormal 1g={eth['anormal_getiri_1g']} isabet={eth['isabet_1g']}")

    ist = ev.get_summary_stats(path=target, min_dilim=30)
    assert ist["genel"]["yetersiz"] is True and ist["genel"]["isabet_1g"] is None
    assert "canli" in ist["veri_tipi"]
    print("[OK] istatistik: 30'dan az tamamlanmış tahminde 'yetersiz' dönüyor")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_migrasyon(tmp)
        test_alinti_kanit_zorunlu(tmp)
        test_degistirilemezlik(tmp)
        test_referans()
        test_isabet()
        test_10_tahmin_ve_degerlendirme(tmp)
    print("\nTÜM TRACKER TESTLERİ GEÇTİ")


if __name__ == "__main__":
    main()
