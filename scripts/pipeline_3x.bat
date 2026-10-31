@echo off
REM ============================================================================
REM Agora - gunde 3 kez canli hat (toplama + etiketleme + cerceve + piyasa cikarimi)
REM LLM kullanir: bagli her calistirma DeepSeek faturasi olusturur.
REM Kurulum: gorev zamanlayiciya ekleyin (asagidaki talimata bakin).
REM ============================================================================
setlocal
cd /d "%~dp0.."
if not exist data\logs_pipeline.log echo. > data\logs_pipeline.log
echo [%date% %time%] Canli hat basliyor... >> data\logs_pipeline.log
python -X utf8 main.py --cluster-mode embedding --market >> data\logs_pipeline.log 2>&1
echo [%date% %time%] Bitti (cikis kodu: %errorlevel%) >> data\logs_pipeline.log
endlocal
