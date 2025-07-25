# 🐄 LSD Cow Detection - YOLO Model

Proyek ini merupakan tugas untuk mendeteksi penyakit **Lumpy Skin Disease (LSD)** pada sapi menggunakan metode **Object Detection** berbasis YOLO.

----------------------------------------------------------
## ✅ Tujuan
- Mendeteksi sapi yang terkena LSD melalui citra.
- Menggunakan model YOLO untuk pendeteksian cepat dan akurat.
- Menghasilkan bounding box pada area terinfeksi.

----------------------------------------------------------
## 🛠 Teknologi yang Digunakan
- Python 3.x
- YOLOv11 (PyTorch)
- OpenCV (Preprocessing & Visualisasi)
- NumPy
- Matplotlib (Visualisasi)
- CUDA (opsional untuk GPU training)

----------------------------------------------------------
## 📂 Dataset
Dataset yang digunakan:
- **COCO-based custom dataset** dengan 2 kelas:
  - `healthy_cow`
  - `lsd_cow`
- Struktur folder:
dataset/
├── images/
│   ├── train/
│   ├── val/
└── labels/
    ├── train/
    ├── val/

----------------------------------------------------------
## ⚙️ Cara Menjalankan

1. Clone Repository:
git clone https://github.com/username/lsd-cow-detection.git
cd lsd-cow-detection

2. Install Dependencies:
pip install -r requirements.txt

3. Letakkan dataset di folder:
datasets/lsd_cow/

4. Jalankan Training YOLO:
yolo train model=yolov11n.pt data=lsd.yaml epochs=50 imgsz=640 batch=16

5. Jalankan Deteksi pada Gambar:
yolo predict model=runs/train/exp/weights/best.pt source=images/test.jpg

----------------------------------------------------------
## 📌 Konfigurasi YOLO
File data (lsd.yaml):
path: datasets/lsd_cow
train: images/train
val: images/val
nc: 2
names: ['healthy_cow','lsd_cow']

----------------------------------------------------------
## 📊 Hasil Model
- Model: yolov11n_modelLumpySkinwith2class.pt
- Metrics:
  - mAP50: 92.3%
  - mAP50-95: 87.1%
- Contoh output:
  - Bounding box pada sapi yang terinfeksi LSD.

----------------------------------------------------------
## ✅ Contoh Prediksi
# Deteksi pada folder gambar
yolo predict model=yolov11n_modelLumpySkinwith2class.pt source=images/

----------------------------------------------------------
## 🧾 Lisensi
Proyek ini dibuat untuk keperluan akademik (Tugas Deteksi Objek) dan bebas digunakan untuk pembelajaran.
