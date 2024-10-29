1.	Proje Genel Bakış

Bu proje makine öğrenmesi ve görüntü işleme teknikleri kullanılarak elma yapraklarındaki bazı hastalıkları tespit etmeyi amaçlayan bir sistem geliştirmeyi amaçlamaktadır. Sistem, görüntü işleme ve çeşitli makine öğrenmesi algoritmaları kullanarak dört farklı elma yaprağı durumunu sınıflandırabilmektedir.

•	Sağlıklı (Healthy)
•	Elma Uyuzu (Apple Scab)
•	Siyah Çürük (Black Rot)
•	Sedir Elma Pası (Cedar Apple Rust)



2.	 Sistem Gereksinimleri

2.1 Yazılım Gereksinimleri

•	Python 3.x
•	OpenCV (cv2)
•	Pyspark
•	Scikit-learn
•	XGBoost
•	Numpy
•	Pandas
•	JDK 8 (Spark için)

2.2 Donanım Gereksinimleri

•	Google Colab TPU (verimli olması açısından)
•	Yeterli Disk Alanı


3.	Veri Seti

3.1 Veri Seti Yapısı

	Veri seti dört ana kategoride elma yaprağı görüntülerinden oluşmaktadır:
/Apple/
    ├── Apple healthy/
    ├── Apple Apple scab (elma uyuzu)/
    ├── Apple Black rot (siyah curuk)/
    └── Apple Cedar apple rust (sedir elma pasi)/

3.2 Veri Artıma Teknikleri

•	Rotasyon (90°, -90°, 180°)
•	Gri Tonlama Dönüşümü





4.	Teknik Mimari

4.1	Veri İşleme Pipeline’ı

1.	Görüntü Yükleme ve Ön işleme
2.	Veri Artırma
3.	Özellik Çıkarma
4.	Model Eğitimi ve Değerlendirme

4.2 Kullanılan Teknolojiler
•	PySpark : Dağıtık veri işleme
•	OpenCV: Görüntü İşleme
•	Scikit-learn: Model eğitimi ve değerlendirme

5. Uygulama Bileşenleri
5.1	Data Processing (data_processing.py)

•	Veri artırma işlemleri
•	Görüntü rotasyonları
•	Gri Tonlama dönüşümleri
•	CSV etiket dosyaları oluşturma
•	Train/test split işlemleri








5.2	Feature Extraction (feature_extracting.py)

Çıkarılan özellikler:

1.	Renk Özellikleri:
•	BGR renk uzayı ortalamaları
•	HSV renk uzayı ortalamaları

2.	Doku Özellikleri (GLCM):

•	Kontrast
•	Farklılık
•	Homojenlik
•	Enerji
•	Korelasyon

3.	Kenar Özellikleri

•	Kenar yoğunluğu (Canny edge detection)

5.3	Model Selection (ml_model_select.py)

Test edilen modeller:

1.	Random Forest
2.	SVM
3.	XGBoost
4.	KNN



6. Model Performans Analizi
	
Her model için accuracy ve classification report metrikleri hesaplanmıştır, Random Forest, Support Vector Machine, XGBoost ve KNN modelleri eğitildi, KNN ve XGBoost modelleri neredeyse %100 perfomans verdi olarak gözükmesine rağmen, dışardan girilen yeni verileri teste tabi tutma ve cross validation yöntemleri ile overfitting olduğu anlaşıldı SVM modelinin başarı oranı %67 gibidüşük bir orandı. KNN algoritması ise %98'e kadar doğruluk sağladı ve denenen en uygun modelin bu olduğuna karar verildi.


