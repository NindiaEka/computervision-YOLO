# Computer Vision Framework

Framework ini adalah Computer Vision Framework berbasis Python yang saat ini mendukung dua pipeline utama:

1. Training Pipeline untuk workflow dataset hingga model export.
2. Video Processing Pipeline untuk workflow video ke frame extraction beserta summary artefak run.

Framework mempertahankan pola arsitektur yang konsisten:

- Clean Architecture
- Dependency Injection
- Scripts as entry points only
- Business logic di service dan pipeline

## Arsitektur

### Pola Umum

Setiap flow dijalankan dengan pola berikut:

1. scripts/* sebagai CLI entry point tipis.
2. pipelines/* sebagai orchestrator urutan proses.
3. services/* sebagai tempat business logic.
4. utils/* sebagai shared utility (config dan logging).

Dengan pola ini, logic tetap modular, testable, dan mudah diperluas tanpa memengaruhi pipeline yang sudah stabil.

### Training Pipeline

Alur training:

1. ConfigLoader memuat konfigurasi.
2. RoboflowService menyiapkan dataset.
3. DatasetService memvalidasi dataset.
4. TrainingService menjalankan training dan validation YOLO.
5. ExportService menyalin best model ke trained_models.

### Video Processing Pipeline

Alur video processing:

1. ConfigLoader memuat konfigurasi.
2. VideoProcessingPipeline membuat run ID.
3. VideoService membaca video dan metadata.
4. FrameExtractionService mengekstrak frame.
5. Pipeline membangun summary, menyimpan summary.json dan config_snapshot.yaml.

## Struktur Folder

```text
computer-vision/
├── configs/
│   └── config.yaml
├── datasets/
├── experiments/
├── pipelines/
│   └── video_processing_pipeline.py
├── pretrained/
├── scripts/
│   ├── extract_frames.py
│   └── train.py
├── services/
│   ├── dataset_service.py
│   ├── export_service.py
│   ├── frame_extraction_service.py
│   ├── roboflow_service.py
│   ├── training_pipeline.py
│   ├── training_service.py
│   └── video_service.py
├── trained_models/
├── utils/
│   ├── config.py
│   └── logger.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Instalasi

1. Clone repository

```bash
git clone <repository-url>
cd computer-vision
```

2. Buat virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependency

```bash
pip install -r requirements.txt
```

Opsional menggunakan UV:

```bash
uv pip install -r requirements.txt
```

## Prasyarat

- Python 3.12+
- PyTorch
- Ultralytics
- Roboflow SDK
- OpenCV (cv2)
- FFmpeg tersedia di PATH (khusus frame extraction)

## Konfigurasi

Semua pipeline menggunakan satu konfigurasi utama:

- configs/config.yaml

Section yang digunakan:

- project
- roboflow
- model
- training
- validation
- output
- video

Contoh section video yang direkomendasikan:

```yaml
video:
  input_dir: "data/videos"
  output_dir: "data/frames"
  mode: "fps"                # fps | interval
  fps: 1
  interval_seconds: 5
  image_format: "jpg"
```

Catatan:

- mode fps menggunakan nilai fps.
- mode interval menggunakan interval_seconds.

## Menjalankan Pipeline

### 1) Training Pipeline

Jalankan dari root project:

```bash
python scripts/train.py
```

Atau:

```bash
uv run python -m scripts.train
```

Output utama:

- experiments/ untuk artefak run training
- trained_models/ untuk model hasil export

### 2) Video Processing Pipeline

Jalankan dari root project:

```bash
python scripts/extract_frames.py
```

Atau:

```bash
uv run python -m scripts.extract_frames
```

Output utama (per run):

```text
data/frames/
└── run_YYYYMMDD_HHMMSS/
    ├── summary.json
    ├── config_snapshot.yaml
    ├── <video_1>/
    │   ├── frame_000001.jpg
    │   └── ...
    └── <video_2>/
        ├── frame_000001.jpg
        └── ...
```

## Ringkasan Tanggung Jawab Komponen

- scripts/train.py
  - Entry point Training Pipeline.
- scripts/extract_frames.py
  - Entry point Video Processing Pipeline.
- services/training_pipeline.py
  - Orchestrator training flow.
- pipelines/video_processing_pipeline.py
  - Orchestrator video processing flow.
- services/video_service.py
  - Discovery dan metadata video input.
- services/frame_extraction_service.py
  - Ekstraksi frame dari video via FFmpeg.

## Troubleshooting Singkat

1. Video processing gagal di tahap ekstraksi
   - Pastikan FFmpeg sudah terpasang dan bisa dipanggil dari terminal.
2. Video tidak ditemukan
   - Pastikan video.input_dir pada config mengarah ke folder yang benar.
3. Training gagal download dataset
   - Periksa roboflow.api_key, workspace, project, dan version.
