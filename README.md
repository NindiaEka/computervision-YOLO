# Computer Vision Framework

Framework ini adalah Computer Vision Framework berbasis Python yang saat ini mendukung empat workflow utama:

1. Training Pipeline untuk workflow dataset hingga model export.
2. Standalone Validation Pipeline untuk evaluasi model .pt tanpa training ulang.
3. Standalone Inference Pipeline untuk prediksi video menggunakan model .pt yang sudah ada.
4. Video Processing Pipeline untuk workflow video ke frame extraction beserta summary artefak run.

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

### Standalone Validation Pipeline

Alur validasi terpisah:

1. ConfigLoader memuat konfigurasi.
2. ValidationPipeline membuat run ID.
3. ValidationService memuat model dan dataset YAML.
4. ValidationService menjalankan evaluasi YOLO (`model.val`).
5. Pipeline menyimpan `summary.json` dan `config_snapshot.yaml`.

### Standalone Inference Pipeline

Alur inference terpisah:

1. ConfigLoader memuat konfigurasi.
2. InferencePipeline membuat run ID.
3. InferenceService memuat model dan daftar video input.
4. InferenceService menjalankan prediksi video (`model.predict`).
5. Pipeline menyimpan `summary.json`, `config_snapshot.yaml`, dan artefak prediksi.

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
│   ├── inference_pipeline.py
│   ├── training_pipeline.py
│   ├── validation_pipeline.py
│   └── video_processing_pipeline.py
├── pretrained/
├── scripts/
│   ├── extract_frames.py
│   ├── inference.py
│   ├── train.py
│   └── validate.py
├── services/
│   ├── dataset_service.py
│   ├── export_service.py
│   ├── frame_extraction_service.py
│   ├── inference_service.py
│   ├── roboflow_service.py
│   ├── training_service.py
│   ├── validation_service.py
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
- inference
- output
- video

Contoh section video yang direkomendasikan:

```yaml
video:
  input_dir: "data/videos"
  output_dir: "data/frames"
  extraction:
    mode: "fps"              # fps | interval
    fps: 1
    interval_seconds: 5
  image_format: "jpg"
  ffmpeg_path: "ffmpeg"
```

Catatan:

- mode `fps` menggunakan `video.extraction.fps`.
- mode `interval` menggunakan `video.extraction.interval_seconds`.
- Standalone validation menggunakan section `validation`.
- Standalone inference menggunakan section `inference`.

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

### 2) Standalone Validation Pipeline

Jalankan dari root project:

```bash
python scripts/validate.py
```

Atau:

```bash
uv run python -m scripts.validate
```

Output utama (per run):

```text
experiments/validation/
└── run_YYYYMMDD_HHMMSS/
  ├── summary.json
  ├── config_snapshot.yaml
  └── validation artifacts dari Ultralytics (plots/csv jika aktif)
```

### 3) Standalone Inference Pipeline

Jalankan dari root project:

```bash
python scripts/inference.py
```

Atau:

```bash
uv run python -m scripts.inference
```

Output utama (per run):

```text
experiments/inference/
└── run_YYYYMMDD_HHMMSS/
  ├── summary.json
  ├── config_snapshot.yaml
  └── predictions/
    ├── <video_1>.mp4
    └── <video_2>.mp4
```

### 4) Video Processing Pipeline

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
- scripts/validate.py
  - Entry point Standalone Validation Pipeline.
- scripts/inference.py
  - Entry point Standalone Inference Pipeline.
- scripts/extract_frames.py
  - Entry point Video Processing Pipeline.
- pipelines/training_pipeline.py
  - Orchestrator training flow.
- pipelines/validation_pipeline.py
  - Orchestrator standalone validation flow.
- pipelines/inference_pipeline.py
  - Orchestrator standalone inference flow.
- pipelines/video_processing_pipeline.py
  - Orchestrator video processing flow.
- services/validation_service.py
  - Business logic validasi model terhadap dataset.
- services/inference_service.py
  - Business logic inference video dengan model YOLO.
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
4. Validation gagal membaca model/dataset
  - Periksa validation.model_path dan validation.dataset_yaml_path.
5. Inference gagal menemukan video input
  - Pastikan inference.input_dir berisi file video (mp4/avi/mov/mkv).
