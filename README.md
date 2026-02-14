#  Hazır Modellerle Video Analizi Takip (Tracking) + Olay Mantığı (Counting) Yaklaşım
 

MOT17 açık veri seti üzerinde hazır derin öğrenme modelleri kullanarak nesne takibi (tracking) ve belirlenen sanal çizgiler üzerinden kişi sayımı (counting) gerçekleştirmek amacıyla geliştirilmiştir1. Proje, özellikle kalabalık sahnelerdeki örtüşme (occlusion) ve ID değişimi (ID switch) gibi zorlukları aşmaya odaklanan post-processing mekanizmaları içerir.


## Proje Özeti

Sistem, YOLO11 nesne algılama modellerini BoT-SORT ve ByteTrack algoritmalarıyla entegre ederek iki farklı kombinasyon sunar:BoT-SORT + ID Stitching: Düşük ID switch ve yüksek takip kararlılığı için optimize edilmiştir.ByteTrack: Yüksek algılama hassasiyeti ve hızlı CPU çıkarımı için varsayılan yapılandırmada sunulmuştur.## Kullanılan Veri Seti Sekansları

Çalışmada, sistemin farklı senaryolardaki dayanıklılığını test etmek amacıyla MOT17 veri setinden üç farklı zorluk seviyesine sahip sekans seçilmiştir.

| Sekans | Kamera Durumu | Zorluk Seviyesi | Karakteristik |
| :--- | :---: | :---: | :--- |
| MOT17-04 | Sabit | Orta | Belirgin perspektif, kalabalık insan akışı |
| MOT17-09 | Sabit | Orta | Kalabalık sahneler ve yoğun etkileşim |
| MOT17-13 | Hareketli | Zor | Kamera sarsıntısı, yoğun örtüşme (occlusion) ve karmaşık arka plan |

## Kurulum ve Çalıştırma

## Önemli Not – Veri Yerleşimi

MOT17 sekanslarına ait görüntü dosyaları, aşağıdaki dizin yapısına uygun olacak şekilde
**manuel olarak yerleştirilmelidir**:

```text
project_root/
└── data/
    ├── MOT17-04/
    │   └── img1/
    │       ├── 000001.jpg
    │       ├── 000002.jpg
    │       └── ...
    ├── MOT17-09/
    │   └── img1/
    └── MOT17-13/
        └── img1/
```

Proje, CPU üzerinde makul sürede çalışacak şekilde optimize edilmiştir.

``` bash
git clone <repo-url>
cd <repo-folder>


Gerekli kütüphaneleri yükleyin.

``` bash
pip install -r requirements.txt

python src/run.py
```

## Girdi Tanımı ve Sayım Mantığı

Sayım işlemi, configs/lines.yaml dosyasında tanımlanan koordinatlar üzerinden geometrik kesişim analiziyle yapılır.Konumlandırma: Sekansın akış yönüne göre dikey veya yatay sanal çizgiler belirlenmiştir.Mantık: Her bir nesnenin iki kare arasındaki hareket vektörü, tanımlanan çizgi segmentiyle kesiştiğinde vektörel çarpım (cross-product) yöntemiyle yön tayini yapılır.Buffer Mekanizması: Aynı kişinin kısa sürede tekrar sayılmasını önlemek için min_frames_between_crossings: 30 parametresiyle bir tampon süresi uygulanmıştır.
## Çıktı ve Klasör Yapısı

Sistem, her bir sekans ve model kombinasyonu için sonuçları otomatik olarak outputs/ klasörü altında organize eder.

```

outputs/
├── nano+botsort/                 # YOLOv8-nano + BoT-SORT kombinasyonu
│   ├── MOT17-04/
│   │   ├── tracks.txt            # MOT formatında takip çıktıları
│   │   ├── events.json           # Giriş / çıkış ve olay logları
│   │   ├── MOT17-04_demo.mp4     # Overlay edilmiş demo video
│   │   └── summary.png           # Sayısal özet (ID sayısı, FPS vb.)
│   ├── MOT17-09/
│   │   └── ...
│   └── MOT17-13/
│       └── ...
│
├── small+bytetrack/              # YOLOv8-small + ByteTrack kombinasyonu
│   ├── MOT17-04/
│   │   └── ...
│   └── ...
│
├── demo_gifs/                    # Kısa demo GIF çıktıları
│   ├── MOT17-04_demo.gif
│   ├── MOT17-09_demo.gif
│   └── MOT17-13_demo.gif
│
└── overall_summary.json          # Tüm modellerin ve sekansların
                                  # karşılaştırmalı genel özeti
```
