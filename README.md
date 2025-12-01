# 🩺 Cilt Kanseri Tespiti ve Sınıflandırma (Skin Cancer Detection)

![Proje Kapak Görseli](BURAYA_KAPAK_FOTOGRAFI_LINKI_GELECEK_VEYA_DOSYA_YOLU)

> **"Erken teşhis hayat kurtarır."** > Bu proje, yapay zeka destekli görüntü işleme teknikleri kullanarak cilt lezyonlarını analiz eder ve kanser riskini tahmin eder.

## 📖 Proje Hakkında

Bu çalışma, dermatoskopik görüntüleri analiz ederek cilt lezyonlarını **İyi Huylu (Benign)** veya **Kötü Huylu (Malignant)** olarak sınıflandıran, yüksek doğruluk oranına sahip bir Derin Öğrenme (Deep Learning) modelidir.

Projede, görüntü sınıflandırma alanında başarısı kanıtlanmış **Transfer Learning (Xception)** mimarisi kullanılmış ve sınıflandırma başarısını maksimize etmek için **XGBoost** algoritması ile hibrit bir yapı denenmiştir.

### 🎯 Projenin Amacı
* Cilt kanseri teşhisinde doktorlara yardımcı olabilecek bir **Klinik Karar Destek Sistemi** oluşturmak.
* İyi huylu ve kötü huylu lezyonlar arasındaki ince farkları makine öğrenmesi ile ayırt etmek.
* Manuel teşhis hatalarını minimize etmek.

---

## 🛠 Kullanılan Teknolojiler ve Yöntemler

Proje geliştirme sürecinde aşağıdaki modern kütüphaneler ve teknikler kullanılmıştır:

| Teknoloji | Açıklama |
| :--- | :--- |
| **Python** | Ana programlama dili. |
| **TensorFlow / Keras** | Derin öğrenme modeli (CNN & Transfer Learning) eğitimi için. |
| **Xception** | Özellik çıkarımı (Feature Extraction) için kullanılan ön eğitimli model. |
| **XGBoost** | Çıkarılan öznitelikleri sınıflandırarak hassasiyeti artırmak için. |
| **Scikit-learn** | Veri ön işleme, metrik hesaplama ve Confusion Matrix için. |
| **Pandas & NumPy** | Veri manipülasyonu ve matris işlemleri. |
| **Matplotlib & Seaborn** | Veri görselleştirme ve başarı grafikleri. |

![Model Mimarisi veya Eğitim Grafiği](BURAYA_EGITIM_GRAFIGI_LINKI_GELECEK)

---

## 📊 Veri Seti (Dataset)

⚠️ **ÖNEMLİ:** Veri seti dosya boyutu (GB) nedeniyle bu GitHub deposuna **yüklenmemiştir**.

Model, Kaggle üzerindeki **"Skin Cancer: Malignant vs. Benign"** veri seti ile eğitilmiştir.

* **Veri Kaynağı:** [Kaggle - Skin Cancer Dataset](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign)
* **Sınıflar:**
    * `Benign`: İyi huylu, zararsız lezyonlar.
    * `Malignant`: Kötü huylu, kanser riski taşıyan lezyonlar.

### 📂 Klasör Yapısı
Projeyi hatasız çalıştırmak için veriyi indirdikten sonra aşağıdaki dosya yapısını kurunuz:

```text
root/
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
├── malignant-vs-benign-detection.ipynb  # Eğitim Dosyası
├── Test.ipynb                           # Test Dosyası
└── model.h5 (varsa)                     # Eğitilmiş Model
