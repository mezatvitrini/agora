# Zamanlanmış görevler (kurulum talimatı — bu komutlar ÇALIŞTIRILMADI)

İki `.bat` dosyası `scripts/` altında hazır. Kurmak için **yönetici** komut isteminde:

## 1) Salı sabahı değerlendirme (LLM yok, ücretsiz)

```bat
schtasks /Create /TN "Agora-Evaluate-Tuesday" ^
  /TR "\"C:\Users\Kıvanç\PycharmProjects\agora\scripts\evaluate_tuesday.bat\"" ^
  /SC WEEKLY /D TUE /ST 09:00 /F
```

- Ne yapar: `python -m tracker.evaluator` → `beklemede` kayıtların 1g/7g getirisini ve isabetini doldurur.
- Log: `data\logs_evaluator.log`
- Elle test: `scripts\evaluate_tuesday.bat`

## 2) Günde 3 kez canlı hat (LLM kullanır — maliyetlidir)

```bat
schtasks /Create /TN "Agora-Pipeline-3x" ^
  /TR "\"C:\Users\Kıvanç\PycharmProjects\agora\scripts\pipeline_3x.bat\"" ^
  /SC DAILY /ST 08:00 /RI 480 /DU 24:00 /F
```

- `/RI 480` = 480 dakikada bir (günde 3 kez: 08:00, 16:00, 00:00).
- Ne yapar: `python main.py --cluster-mode embedding --market`
  (RSS toplama → etiketleme → çerçeve karşılaştırması → alıntı doğrulama → canlı piyasa çıkarımı → değerlendirme).
- Log: `data\logs_pipeline.log`
- Maliyet notu: her çalıştırma yeni başlıklar için etiketleme çağrıları + çerçeve çağrıları + piyasa çıkarımı yapar.
  Günlük 3 çalıştırma, yeni haber hacmine göre ~$0,05-0,30/gün.

## Kontrol / kaldırma

```bat
schtasks /Query /TN "Agora-Evaluate-Tuesday" /V /FO LIST
schtasks /Query /TN "Agora-Pipeline-3x" /V /FO LIST
schtasks /Delete /TN "Agora-Pipeline-3x" /F
```

## Notlar

- Bilgisayar uykuya girerse görev çalışmaz; `schtasks /Change /TN "Agora-Pipeline-3x" /IT` ile yalnızca oturum açıkken çalışacak şekilde ayarlanabilir.
- GDELT geri tarama hatları bu görevlere **dahil değildir** (rate limit nedeniyle ayrı yürütülür).
