"""Embedding kümeleyici testi (tamamen yerel, API çağrısı yok).

Çalıştırma: python test_embedding_clusterer.py
İlk çalıştırmada model (~470 MB) indirilir; sonrasında embedding önbelleği kullanılır.
"""
from datetime import date

import analyzer.embedding_clusterer as ec
from analyzer.embedding_clusterer import cluster_rows, embed_titles

GUN = date(2026, 10, 30)

# 7 sahte başlık: 2 olay (biri 3 dilli, biri 2 dilli) + 2 alakasız haber
BASLIKLAR = [
    # OLAY 1 — İspanya'da tren kazası (Türkçe, İngilizce, Rusça)
    ("hurriyet", "TR", "tr", "İspanya'daki tren kazasında can kaybı 69'a yükseldi"),
    ("bbc_world", "GB", "en", "Spain train crash death toll rises to 69"),
    ("rt", "RU", "ru", "Число погибших при крушении поезда в Испании выросло до 69"),
    # OLAY 2 — Binance'e siber saldırı (İngilizce, Türkçe)
    ("coindesk", "US", "en", "Binance hacked: attackers move $6.3 billion in crypto assets"),
    ("aa", "TR", "tr", "Binance'e siber saldırı: 6,3 milyar dolarlık kripto varlık etkilendi"),
    # ALAKASIZ
    ("hurriyet", "TR", "tr", "Galatasaray deplasmanda Konyaspor'u 4-0 yendi"),
    ("nyt_world", "US", "en", "New York subway strike ends after three days"),
]


def sahte_satirlar():
    return [
        {
            "title": baslik,
            "source": kaynak,
            "ulke": ulke,
            "dil": dil,
            "kaynak_tipi": "ozel",
            "day": GUN,
            "categories": ["siyaset"],
        }
        for kaynak, ulke, dil, baslik in BASLIKLAR
    ]


def kume_bul(kumeler, baslik_parcasi):
    for index, kume in enumerate(kumeler, start=1):
        if any(baslik_parcasi.lower() in m["title"].lower() for m in kume):
            return index, {m["title"] for m in kume}
    return None, set()


def main():
    rows = sahte_satirlar()
    kumeler = cluster_rows(rows)

    print(f"{len(rows)} başlık -> {len(kumeler)} küme (eşik {ec.CLUSTER_THRESHOLD})")
    for index, kume in enumerate(kumeler, start=1):
        print(f"  Küme {index}:")
        for m in kume:
            print(f"    - ({m['source']}/{m['dil']}) {m['title']}")

    # 1) Aynı olay, 3 farklı dilde birleşmeli
    tren_id, tren = kume_bul(kumeler, "tren kazasında can kaybı")
    assert tren_id is not None, "İspanya tren kazası kümesi bulunamadı"
    beklenen_tren = {b for _, _, _, b in BASLIKLAR[:3]}
    assert tren == beklenen_tren, f"Tren kazası kümesi eksik/fazla: {tren ^ beklenen_tren}"
    print(f"[OK] Olay 1 (TR+EN+RU) tek kümede: {tren_id}")

    # 2) İkinci olay (EN+TR) ayrı kümede birleşmeli
    binance_id, binance = kume_bul(kumeler, "Binance")
    assert binance_id is not None, "Binance kümesi bulunamadı"
    beklenen_binance = {b for _, _, _, b in BASLIKLAR[3:5]}
    assert binance == beklenen_binance, f"Binance kümesi eksik/fazla: {binance ^ beklenen_binance}"
    assert binance_id != tren_id, "İki farklı olay aynı kümeye düştü"
    print(f"[OK] Olay 2 (EN+TR) ayrı kümede: {binance_id}")

    # 3) Alakasız başlıklar bu kümelere girmemeli
    for _, _, _, alakasiz in BASLIKLAR[5:]:
        assert alakasiz not in tren and alakasiz not in binance, f"Alakasız başlık kümeye girdi: {alakasiz}"
    print("[OK] Alakasız başlıklar olay kümelerine girmedi")

    # 4) Küme boyutu ve tarih sınırları
    asserts = all(len(k) <= ec.CLUSTER_MAX_SIZE for k in kumeler)
    assert asserts, f"Küme boyutu {ec.CLUSTER_MAX_SIZE}'i aştı"
    print(f"[OK] Küme boyutu sınırı ({ec.CLUSTER_MAX_SIZE}) aşılmadı")

    # 5) Embedding önbelleği: ikinci çağrıda model yüklenmemeli
    orijinal = ec.load_model
    ec.load_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("model yeniden yüklendi"))
    try:
        embed_titles([b for _, _, _, b in BASLIKLAR])
        print("[OK] Embedding'ler önbellekten geldi (model yüklenmedi)")
    finally:
        ec.load_model = orijinal

    print("\nTÜM TESTLER GEÇTİ")


if __name__ == "__main__":
    main()
