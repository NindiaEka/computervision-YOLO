from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ultralytics import YOLO

from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Represents inference output for a single input video.

    Attributes:
        input_video (str): Absolute input video path.
        output_video (str): Absolute output video path with predictions.
        duration (float): Input video duration in seconds.
        fps (float): Input video FPS.
        resolution (str): Input video resolution as WIDTHxHEIGHT.
        total_frames (int): Input video total frame count.
        processed_frames (int): Actual processed frame count from inference stream.
        total_detections (int): Total detected objects in this video.
        processing_time (float): Inference duration for this video in seconds.
        inference_fps (float): Effective processed frame speed during inference.
        status (str): Processing status for this video (success or failed).
    """

    input_video: str
    output_video: str
    duration: float
    fps: float
    resolution: str
    total_frames: int
    processed_frames: int
    total_detections: int
    processing_time: float
    inference_fps: float
    status: str


class InferenceService:
    """Service to execute standalone video inference using Ultralytics YOLO.

    This service loads an existing model (.pt), reads all videos in configured
    inference input directory, runs `YOLO.predict()` on each video, and returns
    structured per-video outputs for the pipeline summary.

    Attributes:
        config (Dict[str, Any]): Loaded framework configuration.
        inference_config (Dict[str, Any]): Inference-specific config section.
        output_config (Dict[str, Any]): Output directory configuration section.
        framework_root (Path): Root directory of the framework project.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes InferenceService with dependency-injected config loader.

        Args:
            config_loader (Optional[ConfigLoader]): Config loader instance.
                If None, a default ConfigLoader is created.
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        try:
            self.config = config_loader.get_config()
        except RuntimeError:
            self.config = config_loader.load()

        self.inference_config = self.config.get("inference", {})
        self.output_config = self.config.get("output", {})
        self.framework_root = Path(__file__).resolve().parent.parent

    SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    def infer(
        self,
        run_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> List[InferenceResult]:
        """Runs video inference and returns structured per-video results.

        Args:
            run_id (Optional[str]): Optional run identifier for traceability.
            output_dir (Optional[Path]): Optional output directory provided by
                pipeline. If None, output is derived from config.

        Returns:
            List[InferenceResult]: Structured per-video inference outputs.

        Raises:
            FileNotFoundError: If model or input directory cannot be resolved.
            RuntimeError: If inference execution fails.
        """
        model_path = self._resolve_model_path()
        input_dir = self._resolve_input_dir()
        videos = self._list_input_videos(input_dir)
        run_output_dir = self._resolve_run_output_dir(output_dir, run_id)
        predictions_dir = run_output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting inference using model: %s", model_path)
        logger.info("Using inference input directory: %s", input_dir)
        logger.info("Found %d video(s) for inference.", len(videos))

        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            message = f"Inference process failed: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        results: List[InferenceResult] = []
        total_videos = len(videos)
        for index, video_path in enumerate(videos, start=1):
            logger.info("[%d/%d]", index, total_videos)
            logger.info("Processing %s", video_path.name)
            results.append(
                self._infer_single_video(
                    model=model,
                    video_path=video_path,
                    run_output_dir=run_output_dir,
                    predictions_dir=predictions_dir,
                )
            )

        logger.info("Inference completed for %d video(s).", len(results))
        return results

    def _resolve_model_path(self) -> Path:
        """Resolves inference model path from config.

        Resolution priority:
            1. inference.model_path
            2. latest .pt under output.trained_models_dir

        Returns:
            Path: Absolute model file path.

        Raises:
            FileNotFoundError: If no valid model path can be found.
        """
        configured_model_path = str(self.inference_config.get("model_path", "")).strip()

        if configured_model_path:
            model_path = self._resolve_path(configured_model_path)
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Configured inference model not found: {model_path}"
                )
            return model_path

        trained_models_dir_cfg = str(
            self.output_config.get("trained_models_dir", "trained_models")
        ).strip()
        trained_models_dir = self._resolve_path(trained_models_dir_cfg)

        if not trained_models_dir.is_dir():
            raise FileNotFoundError(
                "Inference model path is not configured and trained models "
                f"directory does not exist: {trained_models_dir}"
            )

        candidate_models = sorted(
            trained_models_dir.glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidate_models:
            raise FileNotFoundError(
                "Inference model path is not configured and no .pt model was "
                f"found in: {trained_models_dir}"
            )

        selected_model = candidate_models[0].resolve()
        logger.info("Using latest trained model for inference: %s", selected_model)
        return selected_model

    def _resolve_input_dir(self) -> Path:
        """Resolves inference input directory from config.

        Returns:
            Path: Absolute input directory path.

        Raises:
            FileNotFoundError: If input directory does not exist.
            NotADirectoryError: If input path is not a directory.
        """
        input_dir_cfg = str(self.inference_config.get("input_dir", "data/inference")).strip()
        if not input_dir_cfg:
            raise FileNotFoundError("Inference input_dir is empty in config section 'inference'.")

        input_dir = self._resolve_path(input_dir_cfg)
        if not input_dir.exists():
            raise FileNotFoundError(f"Configured inference input directory not found: {input_dir}")

        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"Configured inference input path is not a directory: {input_dir}"
            )

        return input_dir

    def _list_input_videos(self, input_dir: Path) -> List[Path]:
        """Lists supported input videos from the configured input directory.

        Args:
            input_dir (Path): Input directory containing inference videos.

        Returns:
            List[Path]: Sorted video file list.

        Raises:
            RuntimeError: If no supported video files are found.
        """
        video_files = [
            file
            for file in input_dir.iterdir()
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_VIDEO_EXTENSIONS
        ]

        if not video_files:
            raise RuntimeError(
                "No supported video files found in inference input directory: "
                f"{input_dir}"
            )

        return sorted(video_files, key=lambda file: file.name.lower())

    def _resolve_run_output_dir(self, output_dir: Optional[Path], run_id: Optional[str]) -> Path:
        """Resolves run output directory with pipeline override support.

        Args:
            output_dir (Optional[Path]): Optional output directory from pipeline.
            run_id (Optional[str]): Optional run identifier.

        Returns:
            Path: Resolved run output directory.
        """
        if output_dir is not None:
            return output_dir.resolve()

        output_root = self._resolve_path(
            str(self.inference_config.get("output_dir", "experiments/inference")).strip()
        )
        run_name = str(run_id or self.inference_config.get("run_name", "inference")).strip()
        if not run_name:
            run_name = "inference"

        return (output_root / run_name).resolve()

    def _build_predict_kwargs(self, source_value: str, run_output_dir: Path) -> Dict[str, Any]:
        """Builds kwargs for Ultralytics YOLO inference.

        Args:
            source_value (str): Inference source value.
            run_output_dir (Path): Run output directory from pipeline.

        Returns:
            Dict[str, Any]: Ultralytics model.predict() keyword arguments.
        """
        return {
            "source": source_value,
            "conf": float(self.inference_config.get("conf", 0.25)),
            "iou": float(self.inference_config.get("iou", 0.6)),
            "imgsz": int(self.inference_config.get("imgsz", 640)),
            "max_det": int(self.inference_config.get("max_det", 300)),
            "device": str(self.inference_config.get("device", "")).strip(),
            "half": bool(self.inference_config.get("half", False)),
            "save": True,
            "save_txt": bool(self.inference_config.get("save_txt", False)),
            "save_conf": bool(self.inference_config.get("save_conf", False)),
            "project": str(run_output_dir),
            "name": "predictions",
            "exist_ok": True,
            "verbose": bool(self.inference_config.get("verbose", False)),
            "stream": True,
        }

    def _infer_single_video(
        self,
        model: YOLO,
        video_path: Path,
        run_output_dir: Path,
        predictions_dir: Path,
    ) -> InferenceResult:
        """Runs inference for a single video and returns structured result.

        Args:
            model (YOLO): Loaded YOLO model instance.
            video_path (Path): Input video path.
            run_output_dir (Path): Run output directory.
            predictions_dir (Path): Prediction output directory.

        Returns:
            InferenceResult: Per-video inference result.
        """
        fps, frame_count, width, height = self._read_video_metadata(video_path)
        total_frames = frame_count
        duration = round((frame_count / fps), 4) if fps > 0 else 0.0
        resolution = f"{width}x{height}"

        started = time.perf_counter()
        status = "success"
        output_video = ""
        total_detections = 0
        processed_frames = 0
        last_logged_progress = 0.0
        pre_existing_files = self._list_prediction_video_files(predictions_dir)

        logger.info("=" * 40)
        logger.info("Processing %s", video_path.name)
        logger.info("=" * 40)

        try:
            predict_kwargs = self._build_predict_kwargs(str(video_path), run_output_dir)
            predict_stream = model.predict(**predict_kwargs)

            for result in predict_stream:
                processed_frames += 1
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    total_detections += int(len(boxes))

                current_progress = (
                    (processed_frames / frame_count) * 100.0 if frame_count > 0 else 0.0
                )

                should_log = (current_progress - last_logged_progress) >= 5.0

                if should_log:
                    self._log_inference_progress(
                        processed_frames=processed_frames,
                        total_frames=frame_count,
                    )
                    last_logged_progress = current_progress

            output_video = self._resolve_output_video_path(
                predictions_dir,
                video_path,
                pre_existing_files,
            )

            if not output_video:
                status = "failed"
                logger.error(
                    "Inference output video not found for '%s' in '%s'.",
                    video_path.name,
                    predictions_dir,
                )
        except Exception as exc:
            status = "failed"
            logger.error("Inference failed for '%s': %s", video_path.name, exc)

        processing_time = round(time.perf_counter() - started, 4)
        inference_fps = round(
            (processed_frames / processing_time) if processing_time > 0 else 0.0,
            4,
        )

        # Ensure final progress snapshot is visible even when periodic threshold
        # was not reached near the end of processing.
        if status == "success":
            self._log_inference_progress(
                processed_frames=processed_frames,
                total_frames=frame_count,
            )

        return InferenceResult(
            input_video=str(video_path.resolve()),
            output_video=output_video,
            duration=duration,
            fps=round(fps, 4),
            resolution=resolution,
            total_frames=total_frames,
            processed_frames=processed_frames,
            total_detections=total_detections,
            processing_time=processing_time,
            inference_fps=inference_fps,
            status=status,
        )

    def _read_video_metadata(self, video_path: Path) -> Tuple[float, int, int, int]:
        """Reads basic metadata from a video file using OpenCV.

        Args:
            video_path (Path): Input video path.

        Returns:
            Tuple[float, int, int, int]: FPS, frame count, width, height.

        Raises:
            RuntimeError: If video file cannot be opened.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video for metadata: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

        return fps, frame_count, width, height

    def _resolve_output_video_path(
        self,
        predictions_dir: Path,
        input_video_path: Path,
        pre_existing_files: Set[Path],
    ) -> str:
        """Resolves output video path generated by YOLO prediction.

        Resolution priority:
            1. Same filename as input video.
            2. Filename with suffix based on input stem.
            3. Newest output video file in predictions directory.

        Args:
            predictions_dir (Path): Prediction output directory.
            input_video_path (Path): Input video path.
            pre_existing_files (Set[Path]): Existing prediction files before
                current video inference started.

        Returns:
            str: Absolute output video path, or empty string if not found.
        """
        if not predictions_dir.is_dir():
            return ""

        expected = predictions_dir / input_video_path.name
        if expected.is_file():
            return str(expected.resolve())

        pattern = f"{input_video_path.stem}*{input_video_path.suffix}"
        matches = sorted(
            predictions_dir.glob(pattern),
            key=lambda file: file.stat().st_mtime,
            reverse=True,
        )

        if matches:
            return str(matches[0].resolve())

        current_files = self._list_prediction_video_files(predictions_dir)
        new_files = [
            file for file in current_files if file.resolve() not in pre_existing_files
        ]
        if new_files:
            newest_new = max(new_files, key=lambda file: file.stat().st_mtime)
            return str(newest_new.resolve())

        if current_files:
            newest_any = max(current_files, key=lambda file: file.stat().st_mtime)
            return str(newest_any.resolve())

        return ""

    def _log_inference_progress(
        self,
        processed_frames: int,
        total_frames: int,
    ) -> None:
        """Logs periodic inference progress using bar and frame counters.

        Args:
            processed_frames (int): Number of processed frames so far.
            total_frames (int): Total frames reported by input metadata.
        """
        progress_bar = self._build_progress_bar(
            processed_frames=processed_frames,
            total_frames=total_frames,
        )
        logger.info("%s", progress_bar)

        if total_frames > 0:
            frame_line = f"{processed_frames} / {total_frames}"
        else:
            frame_line = f"{processed_frames} / N/A"

        logger.info("Frame    : %s", frame_line)

    def _build_progress_bar(
        self,
        processed_frames: int,
        total_frames: int,
        width: int = 20,
    ) -> str:
        """Builds a textual progress bar for inference progress.

        Args:
            processed_frames (int): Number of processed frames.
            total_frames (int): Total frame count from metadata.
            width (int): Character width of the bar.

        Returns:
            str: Formatted progress bar string.
        """
        if total_frames <= 0:
            ratio = 0.0
        else:
            ratio = min(max(processed_frames / total_frames, 0.0), 1.0)

        filled = int(round(ratio * width))
        bar = "#" * filled + "-" * (width - filled)
        percent = int(round(ratio * 100))
        return f"[{bar}] {percent}%"

    def _list_prediction_video_files(self, predictions_dir: Path) -> Set[Path]:
        """Lists output video files currently present in predictions directory.

        Args:
            predictions_dir (Path): Prediction output directory.

        Returns:
            Set[Path]: Set of resolved video file paths.
        """
        if not predictions_dir.is_dir():
            return set()

        return {
            file.resolve()
            for file in predictions_dir.iterdir()
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_VIDEO_EXTENSIONS
        }

    def _resolve_path(self, path_value: str) -> Path:
        """Resolves path relative to framework root when not absolute.

        Args:
            path_value (str): Path string from configuration.

        Returns:
            Path: Resolved absolute path.
        """
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self.framework_root / path
        return path.resolve()
