"""Agora - Streamlit arayüzü."""
import streamlit as st
import pandas as pd
from pathlib import Path

PROCESSED_CSV = Path("data/processed/headlines_tagged.csv")
FRAME_JSON = Path("data/processed/frame_comparison.json")
FRAME_EMBEDDING_JSON = Path("data/processed/frame_comparison_embedding.json")
GAPS_EMBEDDING_CSV = Path("data/processed/gaps_embedding.csv")
PREDICTIONS_CSV = Path("data/predictions.csv")

KAYNAK_TIPI_ETIKET = {
    "devlet_kontrollu": "devlet kontrollü",
    "kamu_yayincisi": "kamu yayıncısı",
    "devlet_finansmanli": "devlet finansmanlı",
    "devlet_baglantili": "devlet kontrollü",  # eski veriler için
    "ozel": "özel",
    "ajans": "ajans",
}

st.set_page_config(page_title="Agora", page_icon="🏛️", layout="wide")

st.title("🏛️ Agora")
st.caption("Haberlerin arkasındaki çerçeveyi ve ayrılıkları görünür kılan araç")

if not PROCESSED_CSV.exists():
    st.error("Veri bulunamadı. Önce `python main.py` çalıştırın.")
    st.stop()

df = pd.read_csv(PROCESSED_CSV)

# --- Yan panel: filtreler ---
st.sidebar.header("Filtreler")
kaynaklar = ["Tümü"] + sorted(df["source"].dropna().unique().tolist())
secili_kaynak = st.sidebar.selectbox("Kaynak", kaynaklar)

# Kategori listesi (JSON string'den çıkar)
import json
def _kat_listesi(s):
    try:
        return json.loads(s) if isinstance(s, str) else []
    except Exception:
        return []

df["kat_listesi"] = df["kategoriler"].apply(_kat_listesi)
tum_kategoriler = sorted({k for lst in df["kat_listesi"] for k in lst})
secili_kategoriler = st.sidebar.multiselect("Kategoriler", tum_kategoriler)

# Filtre uygula
filtreli = df.copy()
if secili_kaynak != "Tümü":
    filtreli = filtreli[filtreli["source"] == secili_kaynak]
if secili_kategoriler:
    filtreli = filtreli[filtreli["kat_listesi"].apply(lambda lst: any(k in lst for k in secili_kategoriler))]


# --- Çerçeveleme karşılaştırması yardımcıları ---
def _liste_metni(items):
    return ", ".join(items) if items else "—"


def _kaynak_panosu(analiz):
    """v2 (çerçeve alanları) ve v1 (seçili kelime/aktif özne) şemalarını birlikte gösterir."""
    baslik = f"**{analiz.get('kaynak', '—')}**"
    if analiz.get("ulke"):
        baslik += f" · {analiz['ulke']}"
    st.markdown(baslik)
    st.caption(analiz.get("baslik", "—"))

    if "cerceve_ozeti" in analiz or "problem_tanimi" in analiz:
        st.markdown(f"**Problem tanımı:** {analiz.get('problem_tanimi') or '—'}")
        st.markdown(f"**Öne çıkan aktör:** {analiz.get('one_cikan_aktor') or '—'}")
        st.markdown(f"**Ton:** {analiz.get('ton') or '—'}")
        st.markdown(f"**Çerçeve:** {analiz.get('cerceve_ozeti') or '—'}")
        dogrulama = analiz.get("alinti_dogrulandi")
        st.caption(f"Alıntı: “{analiz.get('kisa_alinti') or 'belirsiz'}”"
                   + (" ⚠️ doğrulanamadı" if dogrulama == "yok" else ""))
        if dogrulama == "yok":
            st.warning("⚠️ Alıntı bu kaynağın başlığında bulunamadı (kaynak metne dayanmıyor olabilir).")
        elif dogrulama == "kisa":
            st.caption("Kısa alıntı — doğrulama için yeterli uzunlukta değil.")
    else:
        st.markdown(f"**Aktif özne:** {analiz.get('aktif_ozne') or '—'}")
        st.markdown(f"**Seçili kelimeler:** {_liste_metni(analiz.get('secili_kelimeler'))}")
        st.markdown("**Verilen bilgiler**")
        for bilgi in analiz.get("verilen_bilgiler") or ["—"]:
            st.markdown(f"- {bilgi}")
        st.markdown("**Atlanan bilgiler**")
        for bilgi in analiz.get("atlanan_bilgiler") or ["—"]:
            st.markdown(f"- {bilgi}")
        if analiz.get("cerceve_notu"):
            st.info(analiz["cerceve_notu"])
    if analiz.get("_llm_atladi"):
        st.caption("Bu haber için model ayrıntılı analiz döndürmedi.")


def _analiz_panolari(karsilastirma):
    analizler = karsilastirma.get("kaynak_analizleri") or []
    for baslangic in range(0, len(analizler), 3):
        parca = analizler[baslangic:baslangic + 3]
        kolonlar = st.columns(len(parca))
        for kolon, analiz in zip(kolonlar, parca):
            with kolon, st.container(border=True):
                _kaynak_panosu(analiz)


def _ortak_ayrisan_farklar(karsilastirma):
    ortak = karsilastirma.get("ortak_bilgiler") or []
    ayrisan = karsilastirma.get("ayrisan_noktalar") or []
    farklar = karsilastirma.get("cerceve_farklari") or []
    if ortak or ayrisan:
        st.markdown("**Ortak bilgiler / ayrışan noktalar**")
        sol, sag = st.columns(2)
        with sol:
            st.markdown("*Ortak bilgiler*")
            for bilgi in ortak or ["—"]:
                st.markdown(f"- {bilgi}")
        with sag:
            st.markdown("*Ayrışan noktalar*")
            for nokta in ayrisan or ["—"]:
                st.markdown(f"- {nokta}")
    if farklar:
        st.markdown("**Çerçeve farkları**")
        for fark in farklar:
            st.markdown(f"- **{fark.get('boyut') or '—'}:** {fark.get('aciklama', '')}")


def _grup_goster(grup):
    karsilastirma = grup.get("karsilastirma") or {}
    haberler = grup.get("haberler") or []
    kaynaklar = grup.get("kaynaklar") or []
    olay = karsilastirma.get("olay") or (haberler[0]["baslik"] if haberler else "Olay grubu")
    etiket = f"{grup.get('grup_id', '?')}. {', '.join(kaynaklar)} · {len(haberler)} haber — {olay[:110]}"
    with st.expander(etiket):
        st.caption(f"Eşleşme: {grup.get('eslesme', '—')} · Kaynak: {len(kaynaklar)}")
        if grup.get("not"):
            st.caption(grup["not"])
        if grup.get("hata"):
            st.warning(f"Karşılaştırma yapılamadı: {grup['hata']}")
        elif karsilastirma.get("ozet"):
            st.success(karsilastirma["ozet"])

        _analiz_panolari(karsilastirma)
        _ortak_ayrisan_farklar(karsilastirma)

        if haberler:
            st.markdown("**Haberler**")
            for haber in haberler:
                st.markdown(
                    f"- [{haber.get('kaynak', '—')}] {haber.get('baslik', '')} — "
                    f"[habere git]({haber.get('link', '')})"
                )


def _kapsama_disi():
    """gaps_embedding.csv: kume_id -> yazmayan ülkeler."""
    if not GAPS_EMBEDDING_CSV.exists():
        return {}
    try:
        gaps = pd.read_csv(GAPS_EMBEDDING_CSV, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    sonuc = {}
    for _, row in gaps.iterrows():
        ulkeler = [u for u in str(row.get("yazmayan_ulkeler", "")).split(";") if u]
        if ulkeler:
            sonuc[row["kume_id"]] = ulkeler
    return sonuc


def _kume_goster(kume, gaps):
    karsilastirma = kume.get("karsilastirma") or {}
    uyumsuz = bool(karsilastirma.get("kume_uyumsuz"))
    haberler = kume.get("haberler") or []
    tipler = ", ".join(KAYNAK_TIPI_ETIKET.get(t, t) for t in kume.get("kaynak_tipleri") or [])
    etiket = (
        f"{kume.get('kume_id', '?')} · {kume.get('ulke_sayisi', '?')} ülke · "
        f"uyum {kume.get('uyum_skoru', '—')} · {len(haberler)} haber"
        + (" · KÜME UYUMSUZ" if uyumsuz else "")
    )
    with st.expander(etiket):
        st.caption(f"Kaynaklar: {', '.join(kume.get('kaynaklar') or []) or '—'} · Tipler: {tipler or '—'}")
        if kume.get("hata"):
            st.warning(f"Karşılaştırma yapılamadı: {kume['hata']}")
        elif uyumsuz:
            ozet = karsilastirma.get("ozet") or "Başlıklar farklı olayları anlatıyor; karşılaştırma yapılmadı."
            st.markdown(
                f"<div style='color:#9aa0a6'>KÜME UYUMSUZ — {ozet}</div>",
                unsafe_allow_html=True,
            )
        elif karsilastirma.get("ozet"):
            st.success(karsilastirma["ozet"])

        _analiz_panolari(karsilastirma)
        _ortak_ayrisan_farklar(karsilastirma)

        kapsama = gaps.get(kume.get("kume_id"), [])
        if kapsama:
            st.markdown(f"**Kapsama dışı (bu kümede yazmayan ülkeler):** {', '.join(kapsama)}")

        if haberler:
            st.markdown("**Haberler**")
            for haber in haberler:
                ozet = haber.get("ozet") or ""
                satir = f"- [{haber.get('kaynak', '—')} | {haber.get('ulke', '—')}] {haber.get('baslik', '')}"
                if ozet:
                    satir += f" — {ozet[:160]}"
                st.markdown(satir)


# --- Tahmin takibi (Tracker) yardımcıları ---
TAHMIN_SUTUNLARI = {
    "tahmin_id": "ID",
    "tahmin_tarihi": "Tarih",
    "haber_basligi": "Haber",
    "varlik": "Varlık",
    "ai_tahmini": "Yön",
    "ai_guven_skoru": "Güven",
    "gerceklesen_getiri_1g": "Anormal getiri 1g (%)",
    "varlik_yon": "Varlık yönü",
    "haber_ton": "Haber tonu",
    "referans_varlik": "Referans",
    "getiri_ham_1g": "Ham 1g (%)",
    "getiri_ham_7g": "Ham 7g (%)",
    "anormal_getiri_1g": "Anormal 1g (%)",
    "anormal_getiri_7g": "Anormal 7g (%)",
    "olcum_tipi": "Ölçüm",
    "isabet_1g": "İsabet 1g",
    "isabet_7g": "İsabet 7g",
    "gerceklesen_getiri_7g": "Anormal getiri 7g (%)",
    "veri_tipi": "Veri tipi",
    "durum": "Durum",
}


def _tahminleri_oku():
    try:
        return pd.read_csv(PREDICTIONS_CSV, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.warning(f"Tahmin dosyası okunamadı: {exc}")
        return None


def _ortalama_getiri(tahminler, kolon):
    degerler = pd.to_numeric(tahminler.get(kolon), errors="coerce").dropna()
    return f"{degerler.mean():.2f}%" if len(degerler) else "—"


def _isabet_metni(tahminler, kolon):
    seri = tahminler[kolon].str.lower() if kolon in tahminler else pd.Series(dtype=str)
    gecerli = seri[seri.isin(["true", "false"])]
    return f"{100 * (gecerli == 'true').mean():.0f}% (n={len(gecerli)})" if len(gecerli) else "—"


# --- Sekmeler ---
def _gri_tablo(satirlar):
    """'yetersiz' işaretli satırları gri gösterir."""
    df = pd.DataFrame(satirlar)
    if "yetersiz" not in df.columns:
        return df
    def _stil(row):
        return ["color: #9aa0a6" if row["yetersiz"] else "" for _ in row]
    return df.style.apply(_stil, axis=1)


def _istatistik_goster():
    try:
        from tracker.evaluator import get_summary_stats
        ist = get_summary_stats()
    except Exception as exc:  # arayüz çökmesin
        st.warning(f"İstatistikler hesaplanamadı: {exc}")
        return
    st.markdown("#### İstatistikler")
    st.caption(f"dilim eşiği: {ist['min_dilim']} tamamlanmış tahmin · toplam {ist['toplam']} · "
               f"tamamlanan {ist['tamamlanan']} · beklemede {ist['beklemede']}")
    basliklar = (("model", "Model"), ("varlik", "Varlık"), ("kategori", "Kategori"),
                 ("guven", "Güven aralığı"), ("veri_tipi", "Veri tipi"))
    for anahtar, baslik in basliklar:
        dilimler = ist.get(anahtar) or {}
        if not dilimler:
            continue
        satirlar = [
            {
                "dilim": ad,
                "n": veri["n"],
                "isabet_1g": veri.get("isabet_1g") if veri.get("isabet_1g") is not None else "yetersiz veri",
                "isabet_7g": veri.get("isabet_7g") if veri.get("isabet_7g") is not None else "yetersiz veri",
                "yetersiz": veri.get("yetersiz", False),
            }
            for ad, veri in dilimler.items()
        ]
        st.markdown(f"*{baslik}*")
        st.dataframe(_gri_tablo(satirlar), width="stretch", hide_index=True)
    baseline = ist.get("baseline_isabet") or {}
    if baseline:
        parca = ", ".join(
            f"{tip}: {'—' if oran is None else round(oran * 100)}% (n={n})"
            for tip, (oran, n) in baseline.items()
        )
        st.caption(f"Baseline (aynı dönemde 'hep yukarı' tahmini): {parca}")


# --- Sekmeler ---
tab_frame, tab_tracker, tab_haber = st.tabs([
    "Çerçeveleme Karşılaştırması",
    "Tahmin Takibi (Tracker)",
    "Haber Akışı",
])

with tab_frame:
    st.markdown("### Çerçeveleme karşılaştırması")
    mod = st.radio("Kaynak", ["Eski gruplar", "Embedding kümeleri"], horizontal=True, key="frame_mod")

    if mod == "Eski gruplar":
        if not FRAME_JSON.exists():
            st.info("Çerçeve karşılaştırması bulunamadı. Önce `python main.py` çalıştırın.")
        else:
            try:
                frame_data = json.loads(FRAME_JSON.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                frame_data = None
                st.warning(f"Çerçeve dosyası okunamadı: {exc}")
            if isinstance(frame_data, dict):
                frame_gruplari = frame_data.get("gruplar") or []
                st.caption(
                    f"{len(frame_gruplari)} olay grubu · model: {frame_data.get('model', '—')} · "
                    f"üretim: {frame_data.get('uretim_zamani', '—')}"
                )
                if frame_data.get("durum") == "api_anahtari_yok":
                    st.warning("LLM anahtarı bulunmadığı için yalnızca gruplama gösteriliyor.")
                elif frame_data.get("durum") == "kismi":
                    st.warning("Bazı gruplar karşılaştırılamadı (LLM hatası).")

                if secili_kaynak != "Tümü" or secili_kategoriler:
                    secili_haberler = set(zip(filtreli["source"], filtreli["title"]))
                    frame_gruplari = [
                        grup for grup in frame_gruplari
                        if any((h.get("kaynak"), h.get("baslik")) in secili_haberler
                               for h in grup.get("haberler") or [])
                    ]
                    if not frame_gruplari:
                        st.info("Seçili filtrelerle eşleşen olay grubu yok. Filtreleri temizleyip tekrar deneyin.")
                if not frame_gruplari:
                    st.info("Karşılaştırılacak olay grubu bulunamadı.")
                for grup in frame_gruplari:
                    _grup_goster(grup)
    else:
        if not FRAME_EMBEDDING_JSON.exists():
            st.info("Embedding karşılaştırması bulunamadı. "
                    "`python main.py --cluster-mode embedding` çalıştırın.")
        else:
            try:
                emb_data = json.loads(FRAME_EMBEDDING_JSON.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                emb_data = None
                st.warning(f"Embedding çıktısı okunamadı: {exc}")
            if isinstance(emb_data, dict):
                kumeler = emb_data.get("kumeler") or []
                istatistik = emb_data.get("istatistik") or {}
                st.caption(
                    f"{len(kumeler)} küme · min ülke: {emb_data.get('min_countries', '—')} · "
                    f"prompt: {emb_data.get('prompt_versiyonu', '—')} · model: {emb_data.get('model', '—')} · "
                    f"maliyet: ${istatistik.get('tahmini_maliyet_usd', 0)}"
                )
                if emb_data.get("durum") == "api_anahtari_yok":
                    st.warning("LLM anahtarı bulunmadığı için yalnızca kümeler gösteriliyor.")
                elif emb_data.get("durum") == "kismi":
                    st.warning("Bazı kümeler karşılaştırılamadı (LLM hatası).")
                hatali = [k for k in kumeler if k.get("basarisiz")]
                if hatali:
                    st.warning("Başarısız kümeler: " + ", ".join(
                        f"{k.get('kume_id')} ({k.get('hata')})" for k in hatali))
                gaps = _kapsama_disi()
                if not kumeler:
                    st.info("Karşılaştırılacak küme bulunamadı.")
                for kume in kumeler:
                    _kume_goster(kume, gaps)

with tab_tracker:
    st.markdown("### Tahmin Takibi (Tracker)")
    st.warning("Paper takip: bu çıkarımlar yatırım tavsiyesi değildir; geri tarama sonuçları sızıntı ve örneklem sınırları nedeniyle ön göstergedir.")
    if not PREDICTIONS_CSV.exists():
        st.info("Henüz tahmin kaydı yok. `data/predictions.csv` ilk tahminle birlikte oluşur.")
    else:
        if st.button("Tahminleri Değerlendir", help="Bekleyen tahminler için 1g/7g anormal getiriyi hesaplar"):
            try:
                from tracker.evaluator import evaluate_predictions
                with st.spinner("Fiyatlar çekiliyor..."):
                    ozet = evaluate_predictions()
                st.success(f"{ozet['tamamlanan']} tahmin değerlendirildi, "
                           f"{ozet['bekleyen']} tanesi beklemede kaldı.")
            except Exception as exc:  # arayüz çökmesin, hatayı kullanıcıya göster
                st.error(f"Değerlendirme başarısız: {exc}")

        tahminler = _tahminleri_oku()
        if tahminler is not None and tahminler.empty:
            st.info("Tahmin dosyası boş. Tracker ilk tahmini kaydettiğinde tablo dolacak.")
        elif tahminler is not None:
            durum = tahminler["durum"].str.strip() if "durum" in tahminler else pd.Series(dtype=str)
            metrikler = st.columns(5)
            metrikler[0].metric("Toplam tahmin", len(tahminler))
            metrikler[1].metric("Tamamlanan", int((durum == "tamamlandi").sum()))
            metrikler[2].metric("Beklemede", int((durum == "beklemede").sum()))
            metrikler[3].metric("İsabet 1g", _isabet_metni(tahminler, "isabet_1g"))
            metrikler[4].metric("İsabet 7g", _isabet_metni(tahminler, "isabet_7g"))
            st.dataframe(tahminler.rename(columns=TAHMIN_SUTUNLARI), width="stretch", hide_index=True)
            _istatistik_goster()

with tab_haber:
    st.subheader(f"{len(filtreli)} haber")

    # Kategori dağılımı
    st.markdown("### Kategori dağılımı")
    if tum_kategoriler:
        kat_sayilari = {k: 0 for k in tum_kategoriler}
        for lst in filtreli["kat_listesi"]:
            for k in lst:
                kat_sayilari[k] += 1
        st.bar_chart(pd.Series(kat_sayilari).sort_values(ascending=False))
    else:
        st.info("Henüz kategori verisi yok.")

    # Haber listesi
    st.markdown("### Haberler")
    for _, row in filtreli.iterrows():
        with st.expander(f"[{row['source']}] {row['title']}"):
            st.write(f"**Kategoriler:** {', '.join(row['kat_listesi']) or '—'}")
            st.write(f"**Aktörler:** {row.get('aktorler', '—')}")
            st.write(f"**Yayın:** {row.get('published', '—')}")
            st.markdown(f"[Habere git]({row['link']})")
