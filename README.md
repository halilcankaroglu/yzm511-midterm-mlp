# YZM511 Ara Sınav Projesi: MLP ile Sınıflandırma

EDA, fold-local ön işleme, manuel ve otomatik hiperparametre araması ve nested cross-validation ile PyTorch MLP.

- **Ders:** YZM511 Fen Bilimleri Enstitüsü - Yapay Zeka Mühendisliği (Tezsiz) - 2026-2027 Güz Dönemi
- **Öğrenci:** Halil Çankaroğlu
- **Öğrenci No:** 264329016 
- **Rapor:** `report.pdf`

## Veri

UCI Machine Learning Repository, Bank Marketing veri setinin yalnızca `bank-additional.csv` dosyası kullanılmıştır (4.119 gözlem, 20 girdi özelliği, ikili hedef `y`). `bank-additional-full.csv` kullanılmamıştır.

- Kaynak: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Atıf: Moro, S., Rita, P. ve Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Dosya `data/bank-additional.csv` yolunda repoya dahildir; ek indirme gerekmez.

## Çalıştırma

### Google Colab

1. Colab'da **Dosya > Not defterini aç > GitHub** sekmesinden bu repoyu açın.
2. Notebook'ları sırayla baştan sona çalıştırın. Her notebook'un ilk hücresi repoyu Colab'a klonlar ve çalışma dizinini repo köküne taşır.

### Yerel Jupyter

```bash
git clone https://github.com/halilcankaroglu/yzm511-midterm-mlp.git
cd yzm511-midterm-mlp
pip install -r requirements.txt
jupyter notebook
```

### Sıra

| # | Notebook | İçerik | Ürettiği dosyalar |
|---|---|---|---|
| 1 | `01_eda.ipynb` | Veri kontrolleri, seed politikası, stratified %80/%20 split, development-only EDA, deney hipotezleri | `configs/seeds.json`, `results/dataset_split.csv`, `results/eda_*.csv`, `figures/eda_*.png` |
| 2 | `02_nested_cv.ipynb` | 12 adayın sabitlenmesi, fold-local ön işleme, MLP, 3×3 nested CV, referans modeller | `configs/*_candidates.json`, `results/*_search_results.csv`, `results/outer_cv_*.csv`, `results/training_logs.csv` |
| 3 | `03_final_test.ipynb` | Tam development üzerinde nihai seçim, 3 seed ile eğitim, kilitli tahminler, test metrikleri, bootstrap, hata analizi | `results/final_*.json/csv`, `results/test_*.csv`, `models/`, `figures/test_*.png` |

Nested CV (108 iç eğitim + 6 dış kat refit) Colab CPU üzerinde yaklaşık 10 dakika sürmüştür. GPU gerekmez.

## Seed politikası

Temel seed, öğrenci numarasının son beş hanesi (29016) ile 2026'nın yan yana yazılmasıyla elde edilmiştir: **290162026**. Alt seed'ler temel seed'e sabit ofset eklenerek türetilir (`configs/seeds.json`).

| Kullanım | Seed |
|---|---|
| Temel | 290162026 |
| Development/test split | 290162027 |
| Outer CV | 290162028 |
| Inner CV (manuel ve otomatik arama için aynı) | 290162029 |
| Early-stop alt kümesi | 290162030 |
| Random search (SGD / Adam / AdamW; `+k`) | 290162031 / 290162032 / 290162033 |
| Ağırlık başlangıcı: ana model | 290162036 |
| Ağırlık başlangıcı: seed kararlılığı | 290162037, 290162038 |
| Bootstrap güven aralıkları | 290162046 |
| Eşleştirilmiş bootstrap (MLP − LogReg) | 290162047 |

## Sabit kararlar

Aşağıdaki kararlar model performansı incelenmeden alınmıştır.

- `duration` görüşme başlamadan bilinmediği için tüm modellerden çıkarılmıştır.
- `unknown` değerleri gerçek eksik değer sayılmamış, ayrı kategori olarak tutulmuştur.
- `pdays = 999` silinmemiştir. Sabit bir kuralla `pdays_kayitli` ikili bayrağına dönüştürülmüş, 999 değerleri fold içinde öğrenilen median ile doldurulmuştur.
- Satır silme yapılmamıştır. Aykırı değer yönetimi, fold içinde öğrenilen %1–%99 yüzdelik clipping seçeneği olarak arama uzayına dahil edilmiştir.
- Tüm ön işleme parametreleri (median, clipping eşikleri, scaler istatistikleri, kategori sözlüğü) yalnızca ilgili eğitim alt kümesinden öğrenilir (`src/preprocessing.py`).
- Birincil metrik Average Precision (AP), sınıflandırma eşiği 0,50'dir. Early stopping ve LR scheduler iç alt doğrulama kümesindeki BCE kaybını izler.

## Kod yapısı

| Yol | İçerik |
|---|---|
| `src/preprocessing.py` | `pdays` kuralı, yüzdelik clipping, `ColumnTransformer` tabanlı fold-local pipeline |
| `src/mlp.py` | PyTorch MLP (tek logit, `BCEWithLogitsLoss`), optimizer seçimi, early stopping, `ReduceLROnPlateau` |
| `src/nested_cv.py` | İç CV ve aday arama |
| `src/metrics.py` | AP, ROC-AUC, Accuracy, Precision, Sensitivity, Specificity, F1, Balanced Accuracy |
| `configs/` | Seed'ler, eğitim ayarları, arama uzayı, M1–M6 ve R1–R6 adayları |
| `results/` | Split, arama sonuçları, eğitim logları, dış kat sonuçları, nihai seçim, kilitli tahminler, test metrikleri |
| `figures/` | Raporda kullanılan, kodla üretilmiş grafikler |
| `models/` | Nihai MLP (3 seed) ve LogReg modelleri ile öğrenilmiş ön işleme durumları |

## Zaman sıralı deney kaydı

Commit geçmişi deney sırasını belgeler:

1. Seed politikası ve split
2. Development-only EDA
3. 12 aday ve arama uzayı, nested CV başlamadan önce
4. Ön işleme, MLP, metrik ve arama kodu, sonuçlardan önce
5. Nested CV sonuçları
6. **Kilit commit'i (`36cdbe6`):** nihai konfigürasyon, model dosyaları ve etiket içermeyen `test_predictions_locked.csv`
7. Test metrikleri, bootstrap, grafikler, hata analizi

Test etiketleri yalnızca kilit commit'inden sonra okunmuştur. Veri seti kamuya açık olduğundan bu işlem kör test garantisi değil, zaman sıralı bir deney kaydıdır.

## Özet sonuçlar

| | AP | ROC-AUC |
|---|---|---|
| Outer CV, MLP manuel arama (n=3) | 0,420 ± 0,032 | 0,763 ± 0,019 |
| Outer CV, MLP otomatik arama (n=3) | 0,394 ± 0,026 | 0,754 ± 0,020 |
| Outer CV, LogReg (n=3) | 0,415 ± 0,023 | 0,762 ± 0,026 |
| Test, MLP M2 ana seed | 0,441 [0,341; 0,554] | 0,788 [0,730; 0,842] |
| Test, LogReg | 0,429 [0,322; 0,545] | 0,784 [0,723; 0,838] |

Test satırlarındaki köşeli parantezler %95 bootstrap güven aralığıdır. Ayrıntılar ve yorumlar `report.pdf` içindedir.

## Çalışma ortamı

- Google Colab, yalnızca CPU (Intel Xeon @ 2.20GHz, 2 çekirdek, 13,3 GB RAM), CUDA yok
- Python 3.13.18
- Kütüphane sürümleri: `requirements.txt`
