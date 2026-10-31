@echo off
REM ============================================================================
REM Agora - Salı sabahi tahmin degerlendirme
REM Ne yapar: data/predictions.csv icindeki 'beklemede' kayitlar icin 1g/7g
REM getirileri ve isabet alanlarini doldurur (LLM cagrisi YOK, ucretsiz).
REM Kurulum: gorev zamanlayiciya ekleyin (asagidaki talimata bakin).
REM ============================================================================
setlocal
cd /d "%~dp0.."
REM --- Hugging Face / embedding ayarlari (embedding kullanilmasa da tutarli kalsin) ---
if not defined HF_HOME set "HF_HOME=D:\hf_cache"
if not defined HF_HUB_OFFLINE set "HF_HUB_OFFLINE=1"
if not defined TRANSFORMERS_OFFLINE set "TRANSFORMERS_OFFLINE=1"
if not defined EMBEDDING_MODEL_PATH set "EMBEDDING_MODEL_PATH=%HF_HOME%\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
echo [%date% %time%] Tahmin degerlendirme basliyor...
python -X utf8 -m tracker.evaluator >> data\logs_evaluator.log 2>&1
echo [%date% %time%] Bitti (cikis kodu: %errorlevel%)
endlocal
