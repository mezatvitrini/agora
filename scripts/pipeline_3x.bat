@echo off
REM ============================================================================
REM Agora - gunde 3 kez canli hat (toplama + etiketleme + cerceve + piyasa cikarimi)
REM LLM kullanir: bagli her calistirma DeepSeek faturasi olusturur.
REM Kurulum: gorev zamanlayiciya ekleyin (asagidaki talimata bakin).
REM ============================================================================
setlocal
cd /d "%~dp0.."
REM --- Hugging Face / embedding: model D: onbelleginden yuklenir (ag erisimi ve askida kalma yok) ---
if not defined HF_HOME set "HF_HOME=D:\hf_cache"
if not defined HF_HUB_OFFLINE set "HF_HUB_OFFLINE=1"
if not defined TRANSFORMERS_OFFLINE set "TRANSFORMERS_OFFLINE=1"
if not defined EMBEDDING_MODEL_PATH set "EMBEDDING_MODEL_PATH=%HF_HOME%\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
if not exist data\logs_pipeline.log echo. > data\logs_pipeline.log
echo [%date% %time%] Canli hat basliyor... >> data\logs_pipeline.log
python -X utf8 main.py --cluster-mode embedding --market >> data\logs_pipeline.log 2>&1
echo [%date% %time%] Bitti (cikis kodu: %errorlevel%) >> data\logs_pipeline.log
endlocal
