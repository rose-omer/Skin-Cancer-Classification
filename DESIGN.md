# Arayüz tasarımı

Kaynak maket: "04 · Makale Figürü" (`tasarimlar/cilt-demosu/project/D04.dc.html`).
Uygulama: `app/style.css` (tokenler `:root` ve `.dark` altında), `app/app.py` (`theme()`, HTML parçaları).

## Dil

- Kâğıt `#ffffff`, mürekkep `#1a1a1a`, ince çizgi `#bdbdbd`, tek mavi vurgu `#2b5c8a`,
  sonuç için tek turuncu `#c56a1b` (karar yazısı `#9a4e0e`).
- Metin Source Serif 4 (`app/fonts`, OFL). Arayüz ve tablo Source Sans 3 (Gradio paketiyle gelir).
  Eş aralıklı yazı sistem yazı tipidir; IBM Plex Mono yüklenmez.
- Köşe yuvarlatma, gölge, gradyan ve süs ikonu yok. Şekil 2 px mürekkep çizgiyle açılır;
  tablolar booktabs.

## Maketten bilinçli sapmalar

- Karanlık tema: maket yalnız açık. Aynı token adlarıyla türetildi; Gradio `.dark` sınıfını ekler.
- Yükleme alanı ve örnek galerisi: maket sabit bir dosya satırı gösterir ("lezyon_01.jpg, JPEG").
  Gerçek demo dosya ister. Onaydan önce alan gizlidir; yerinde kilit notu durur.
- Gizlilik notu: yüklenen görüntünün ne kadar tutulduğunu söyler. Maket yok; kullanıcı hakkı.
- İngilizce uyarı: yalnız kapanışta, Türkçe uyarının altında bir kez.
- Sonuç özetinin altında Türkçe uyarı dipnotu: sonuç görünürken uyarı da görünür kalsın.
- Sınırlar paragrafında dördüncü cümle (tek kaynak, hasta kimliği yok): veri sızıntısı riski.
- Gradio bileşenleri: kaydırıcının sayı kutusu, sıfırlama düğmesi ve uç yazıları; görüntü
  araç çubukları; etiket simgeleri gizlenir. Kaydırıcı 44 px yüksek.

## Gradio notu

Gradio özel CSS'te `@media` içindeki `.gradio-container` seçicisini önekler; kural eşleşmez.
Sayfa boşluğu bu yüzden üst düzey kuralda `clamp()` ile ayarlanır.
