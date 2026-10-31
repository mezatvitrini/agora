"""Geri tarama konuları: dil bazlı GDELT sorguları, izinli varlıklar, tracker durumu.

GDELT DOC 2.0 sorgu sözdizimi: serbest metin + `sourcelang:<dil>` gibi filtreler.
"""
from __future__ import annotations

TOPICS: dict[str, dict] = {
    "iran_savas": {
        "aciklama": "İran-İsrail/ABD gerilimi ve savaş",
        "sorgular": {
            "tr": '("İran" OR Iran) (savaş OR saldırı OR füze OR yaptırım) sourcelang:turkish',
            "en": '(Iran OR "Strait of Hormuz") (war OR strike OR missile OR sanctions) sourcelang:english',
        },
        "varliklar": ["CL=F", "GC=F", "BTC-USD", "^GSPC"],
        "tracker_aktif": True,
    },
    "kripto": {
        "aciklama": "Kripto para piyasası",
        "sorgular": {
            "tr": '(bitcoin OR ethereum OR kripto OR "kripto para" OR stablecoin) sourcelang:turkish',
            "en": '(bitcoin OR ethereum OR crypto OR stablecoin OR "crypto exchange") sourcelang:english',
        },
        "varliklar": ["BTC-USD", "ETH-USD"],
        "tracker_aktif": True,
    },
    "rusya": {
        "aciklama": "Rusya kaynaklı jeopolitik/enerji haberleri",
        "sorgular": {
            "en": '(Russia OR Putin) (sanctions OR oil OR gas OR Ukraine) sourcelang:english',
            "ru": '(Россия OR Путин) (санкции OR нефть OR газ OR Украина) sourcelang:russian',
        },
        "varliklar": ["CL=F", "NG=F", "^GSPC"],
        "tracker_aktif": True,
    },
    "cin": {
        "aciklama": "Çin ekonomisi ve teknoloji",
        "sorgular": {
            "en": '(China OR Beijing) (economy OR chip OR semiconductor OR stimulus) sourcelang:english',
            "zh": '(中国 OR 北京) (经济 OR 芯片 OR 半导体) sourcelang:chinese',
        },
        "varliklar": ["^GSPC", "BTC-USD", "GC=F"],
        "tracker_aktif": True,
    },
    "turkiye_abd": {
        "aciklama": "Türkiye-ABD ilişkileri",
        "sorgular": {
            "tr": '("Türkiye" OR "Türkiye-ABD") (ABD OR Washington) (ilişki OR anlaşma OR yaptırım) sourcelang:turkish',
            "en": '(Turkey OR Türkiye) (US OR Washington) (relations OR sanctions OR deal) sourcelang:english',
        },
        "varliklar": ["TRY=X", "XU100.IS"],
        "tracker_aktif": False,
    },
    "turkiye_rusya": {
        "aciklama": "Türkiye-Rusya ilişkileri",
        "sorgular": {
            "tr": '("Türkiye" OR "Türkiye-Rusya") (Rusya OR Moskova) (ilişki OR anlaşma OR enerji) sourcelang:turkish',
            "en": '(Turkey OR Türkiye) (Russia OR Moscow) (relations OR deal OR energy) sourcelang:english',
        },
        "varliklar": ["TRY=X", "XU100.IS"],
        "tracker_aktif": False,
    },
    "turkiye_arap": {
        "aciklama": "Türkiye-Arap dünyası ilişkileri",
        "sorgular": {
            "tr": '("Türkiye" OR "Türkiye-Arap") (Arap OR Suudi OR BAE OR Mısır) (ilişki OR anlaşma OR yatırım) sourcelang:turkish',
            "en": '(Turkey OR Türkiye) (Arab OR Saudi OR UAE OR Egypt) (relations OR deal OR investment) sourcelang:english',
        },
        "varliklar": ["TRY=X", "XU100.IS"],
        "tracker_aktif": False,
    },
    "turkiye_pakistan": {
        "aciklama": "Türkiye-Pakistan ilişkileri",
        "sorgular": {
            "tr": '("Türkiye" OR "Türkiye-Pakistan") (Pakistan OR İslamabad) (ilişki OR anlaşma OR savunma) sourcelang:turkish',
            "en": '(Turkey OR Türkiye) (Pakistan OR Islamabad) (relations OR deal OR defence) sourcelang:english',
        },
        "varliklar": ["TRY=X", "XU100.IS"],
        "tracker_aktif": False,
    },
}

TRACKER_AKTIF_KONULAR = [ad for ad, veri in TOPICS.items() if veri["tracker_aktif"]]


def konu_varliklari(konu: str) -> list[str]:
    return list(TOPICS.get(konu, {}).get("varliklar") or [])


def tum_varliklar() -> list[str]:
    varliklar: list[str] = []
    for veri in TOPICS.values():
        for varlik in veri["varliklar"]:
            if varlik not in varliklar:
                varliklar.append(varlik)
    return varliklar
