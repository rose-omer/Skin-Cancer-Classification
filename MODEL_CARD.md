# Model kartı: EfficientNetV2-S, dermoskopi benign / malign

> Bu bir eğitim ve araştırma modelidir. Tıbbi teşhis değildir.

Başlıklar CLEAR Derm kontrol listesini izler. Ölçüt alanları eğitim ve
`python -m skinlesion.evaluate --final` sonrasında `reports/metrics.json` dosyasından doldurulur.

## Amaçlanan kullanım

- Dermoskopi görüntüsünde malign olasılığı tahminini gösteren eğitim ve araştırma demosu.
- Değerlendirme yöntemini (sızıntısız bölünme, eşik, kalibrasyon) öğretmek.

## Kapsam dışı kullanım

- Klinik karar, tarama ya da teşhis.
- Telefon ya da makro fotoğrafları (model yalnız dermoskopi görüntüsüyle eğitildi).
- Ticari kullanım (veri lisansı izin vermez).

## Veri

- Kaynak: Kaggle `fanconic/skin-cancer-malignant-vs-benign`, ISIC Archive kökenli.
- Boyut: train 2637, resmî test 660 (360 benign, 300 malign). Görüntüler 224 px.
- Bölünme: train'den sınıf oranını koruyan %15 val (tohum 42). Test yalnız bir kez ölçülür.
- Lisans: veri seti sayfasında belirtilmemiş; açıklama ticari kullanımı yasaklıyor. ISIC: CC-BY-NC.
- Demografi: yaş, cinsiyet, cilt tonu (Fitzpatrick) ve anatomik bölge bilgisi yok.
- Kopya denetimi: sha256 + dHash (Hamming ≤ 4), `reports/dedup.csv`.

## Model

- Omurga: `timm/tf_efficientnetv2_s.in21k_ft_in1k` (21.5M parametre, Apache-2.0).
- Giriş: 224 px, omurganın kendi normalizasyonu.
- Eğitim: 3 epoch yalnız baş, sonra bütün ağ; AdamW + cosine; `pos_weight`; erken durdurma val
  AUROC'a göre.
- Çıktı: tek logit. Olasılık = sigmoid(logit / T). T ve eşik `config.json` içinde saklanır.

## Değerlendirme

| Ölçüt (test, seçilen eşik) | Değer | %95 GA |
|---|---|---|
| AUROC | eğitim bekliyor | — |
| Duyarlılık | eğitim bekliyor | — |
| Özgüllük | eğitim bekliyor | — |
| pAUC (TPR ≥ 0.80) | eğitim bekliyor | — |
| ECE (kalibrasyon sonrası) | eğitim bekliyor | — |

Karşılaştırma: ilk sürüm (Xception linear probe) duyarlılık 0.763, özgüllük 0.864. O değerler
test sızıntısı nedeniyle iyimserdir.

## Sınırlar ve önyargı

- Koyu tenlerde başarım ölçülmedi.
- Hasta kimliği olmadığı için aynı lezyonun farklı kümelere düşmesi engellenemez.
- Test kümesi dengeli; gerçek yaygınlıkta (ISIC 2020'de yaklaşık %1.8) PPV çok düşer.
- Grad-CAM modelin ağırlık verdiği bölgeyi gösterir; tanı kanıtı değildir.

## Uyarı

Bu bir eğitim ve araştırma demosudur. Tıbbi teşhis değildir. Şüpheli bir ben ya da lezyon için
dermatoloğa başvurun.
