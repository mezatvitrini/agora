"""Agora - Streamlit arayüzü."""
import streamlit as st
import pandas as pd
from pathlib import Path

PROCESSED_CSV = Path("data/processed/headlines_tagged.csv")
FRAME_JSON = Path("data/processed/frame_comparison.json")
PREDICTIONS_CSV = Path("data/predictions.csv")

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

# --- Ana ekran ---
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


# --- Çerçeveleme karşılaştırması ---
def _liste_metni(items):
    return ", ".join(items) if items else "—"


def _kaynak_panosu(analiz):
    st.markdown(f"**{analiz.get('kaynak', '—')}**")
    st.caption(analiz.get("baslik", "—"))
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

        analizler = karsilastirma.get("kaynak_analizleri") or []
        for baslangic in range(0, len(analizler), 3):
            parca = analizler[baslangic:baslangic + 3]
            kolonlar = st.columns(len(parca))
            for kolon, analiz in zip(kolonlar, parca):
                with kolon, st.container(border=True):
                    _kaynak_panosu(analiz)

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

        if haberler:
            st.markdown("**Haberler**")
            for haber in haberler:
                st.markdown(
                    f"- [{haber.get('kaynak', '—')}] {haber.get('baslik', '')} — "
                    f"[habere git]({haber.get('link', '')})"
                )


st.markdown("### Çerçeveleme karşılaştırması")
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


# --- Tahmin takibi (Tracker) ---
TAHMIN_SUTUNLARI = {
    "tahmin_id": "ID",
    "tahmin_tarihi": "Tarih",
    "haber_basligi": "Haber",
    "varlik": "Varlık",
    "ai_tahmini": "Yön",
    "ai_guven_skoru": "Güven",
    "gerceklesen_getiri_1g": "Anormal getiri 1g (%)",
    "gerceklesen_getiri_7g": "Anormal getiri 7g (%)",
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


st.markdown("### Tahmin Takibi (Tracker)")
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
        metrikler[3].metric("Ort. anormal getiri (1g)", _ortalama_getiri(tahminler, "gerceklesen_getiri_1g"))
        metrikler[4].metric("Ort. anormal getiri (7g)", _ortalama_getiri(tahminler, "gerceklesen_getiri_7g"))
        st.dataframe(tahminler.rename(columns=TAHMIN_SUTUNLARI), width="stretch", hide_index=True)