# İlk sürüm (legacy)

`malignant-vs-benign-detection.ipynb`: projenin ilk sürümü, çıktılarıyla birlikte. Yalnız
kayıt içindir; yeni hat `src/skinlesion/` altındadır.

Bilinen sorunlar (ayrıntı ana README'de):

- Test kümesi doğrulama ve erken durdurma için de kullanıldı.
- Xception'a `[0, 1]` girdi verildi; omurga `[-1, 1]` bekler.
- Karar eşiği örtük 0.5; kalibrasyon yok.
- XGBoost deneyi başka bir kümede ölçüldü.
- Model kaydedilmedi.
