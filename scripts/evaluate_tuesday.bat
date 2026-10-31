@echo off
REM ============================================================================
REM Agora - Salı sabahi tahmin degerlendirme
REM Ne yapar: data/predictions.csv icindeki 'beklemede' kayitlar icin 1g/7g
REM getirileri ve isabet alanlarini doldurur (LLM cagrisi YOK, ucretsiz).
REM Kurulum: gorev zamanlayiciya ekleyin (asagidaki talimata bakin).
REM ============================================================================
setlocal
cd /d "%~dp0.."
echo [%date% %time%] Tahmin degerlendirme basliyor...
python -X utf8 -m tracker.evaluator >> data\logs_evaluator.log 2>&1
echo [%date% %time%] Bitti (cikis kodu: %errorlevel%)
endlocal
