from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from services.inference_service import InferenceResult, InferenceService
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class InferenceVideoSummary:
    """Represents per-video inference summary output.

    Attributes:
        input_video (str): Absolute input video path.
        output_video (str): Absolute output video path.
        duration (float): Input video duration in seconds.
        fps (float): Input video FPS.
        resolution (str): Input video resolution in WIDTHxHEIGHT format.
        total_frames (int): Input video total frame count.
        processed_frames (int): Actual processed frame count from inference stream.
        total_detections (int): Total detected objects in this video.
        processing_time (float): Inference time for this video in seconds.
        inference_fps (float): Effective processed frame speed for this video.
        status (str): Processing status for this video.
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


@dataclass(frozen=True)
class InferenceSummary:
    """Represents final inference pipeline summary output.

    Attributes:
        run_id (str): Inference run identifier.
        started_at (datetime): Pipeline start timestamp.
        finished_at (datetime): Pipeline finish timestamp.
        processing_time (float): End-to-end elapsed seconds.
        model_name (str): Model file name used in inference.
        model_path (str): Model file path used in inference.
        imgsz (int): Image size used for inference.
        conf (float): Confidence threshold used for inference.
        iou (float): IoU threshold used for inference.
        device (str): Device used for inference.
        total_videos (int): Total processed videos.
        successful_videos (int): Total successful processed videos.
        failed_videos (int): Total failed processed videos.
        total_frames (int): Total input frames across all videos.
        processed_frames (int): Total processed frames across all videos.
        total_detections (int): Total detected objects across all videos.
        inference_fps (float): Effective processed frame speed for the run.
        videos (List[InferenceVideoSummary]): Per-video summary rows.
        status (str): Inference execution status.
        output_folder (str): Inference run output directory.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    processing_time: float
    model_name: str
    model_path: str
    imgsz: int
    conf: float
    iou: float
    device: str
    total_videos: int
    successful_videos: int
    failed_videos: int
    total_frames: int
    processed_frames: int
    total_detections: int
    inference_fps: float
    videos: List[InferenceVideoSummary]
    status: str
    output_folder: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts summary dataclass to JSON-serializable dictionary.

        Returns:
            Dict[str, Any]: Serialized summary dictionary.
        """
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "processing_time": self.processing_time,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "total_videos": self.total_videos,
            "successful_videos": self.successful_videos,
            "failed_videos": self.failed_videos,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_detections": self.total_detections,
            "inference_fps": self.inference_fps,
            "videos": [
                {
                    "input_video": item.input_video,
                    "output_video": item.output_video,
                    "duration": item.duration,
                    "fps": item.fps,
                    "resolution": item.resolution,
                    "total_frames": item.total_frames,
                    "processed_frames": item.processed_frames,
                    "total_detections": item.total_detections,
                    "processing_time": item.processing_time,
                    "inference_fps": item.inference_fps,
                    "status": item.status,
                }
                for item in self.videos
            ],
            "status": self.status,
            "output_folder": self.output_folder,
        }


class InferencePipeline:
    """Orchestrates standalone inference workflow.

    This pipeline only coordinates process order and persistence. Inference
    business logic remains in InferenceService.

    Attributes:
        config_loader (ConfigLoader): Configuration loader dependency.
        config (Dict[str, Any]): Loaded framework configuration.
        inference_service (InferenceService): Inference business service.
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        inference_service: Optional[InferenceService] = None,
    ) -> None:
        """Initializes InferencePipeline using dependency injection.

        Args:
            config_loader (Optional[ConfigLoader]): Config loader instance.
            inference_service (Optional[InferenceService]): Inference service
                instance.
        """
        self.config_loader = config_loader or ConfigLoader()
        self.config = self._load_configuration()
        self.inference_service = inference_service or InferenceService(
            config_loader=self.config_loader
        )

    def run(self) -> InferenceSummary:
        """Executes full inference pipeline and returns typed summary.

        Returns:
            InferenceSummary: Final inference summary.
        """
        self._validate_dependencies()

        run_id, started_at, output_dir = self._prepare_run()
        inference_results = self._run_inference(run_id, output_dir)
        summary = self._build_summary(
            run_id=run_id,
            started_at=started_at,
            output_dir=output_dir,
            inference_results=inference_results,
        )

        self._save_summary(summary, output_dir)
        self._save_config_snapshot(output_dir)
        self._log_summary(summary)

        return summary

    def _validate_dependencies(self) -> None:
        """Validates required pipeline dependencies.

        Raises:
            RuntimeError: If any required dependency is missing or invalid.
        """
        if self.config_loader is None:
            raise RuntimeError("ConfigLoader dependency is not available.")

        if self.inference_service is None:
            raise RuntimeError("InferenceService dependency is not available.")

        if not hasattr(self.inference_service, "infer"):
            raise RuntimeError("InferenceService dependency must expose infer().")

    def _prepare_run(self) -> Tuple[str, datetime, Path]:
        """Prepares run metadata and inference output directory.

        Returns:
            Tuple[str, datetime, Path]: Run ID, start timestamp, and output path.
        """
        started_at = datetime.now()
        run_id = started_at.strftime("run_%Y%m%d_%H%M%S")

        output_root = self._resolve_output_root()
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return run_id, started_at, output_dir

    def _run_inference(self, run_id: str, output_dir: Path) -> List[InferenceResult]:
        """Runs inference process via InferenceService only.

        Args:
            run_id (str): Inference run identifier.
            output_dir (Path): Inference run output folder.

        Returns:
            List[InferenceResult]: Service inference results.
        """
        return self.inference_service.infer(run_id=run_id, output_dir=output_dir)

    def _build_summary(
        self,
        run_id: str,
        started_at: datetime,
        output_dir: Path,
        inference_results: List[InferenceResult],
    ) -> InferenceSummary:
        """Builds inference summary object from service result.

        Args:
            run_id (str): Inference run identifier.
            started_at (datetime): Inference start timestamp.
            output_dir (Path): Inference output folder.
            inference_results (List[InferenceResult]): Results from InferenceService.

        Returns:
            InferenceSummary: Structured inference summary.
        """
        finished_at = datetime.now()
        processing_time = round((finished_at - started_at).total_seconds(), 4)

        video_rows = [
            InferenceVideoSummary(
                input_video=result.input_video,
                output_video=result.output_video,
                duration=result.duration,
                fps=result.fps,
                resolution=result.resolution,
                total_frames=result.total_frames,
                processed_frames=result.processed_frames,
                total_detections=result.total_detections,
                processing_time=result.processing_time,
                inference_fps=result.inference_fps,
                status=result.status,
            )
            for result in inference_results
        ]

        total_videos = len(video_rows)
        successful_videos = sum(1 for item in video_rows if item.status == "success")
        failed_videos = total_videos - successful_videos
        total_frames = sum(item.total_frames for item in video_rows)
        processed_frames = sum(item.processed_frames for item in video_rows)
        total_detections = sum(item.total_detections for item in video_rows)
        inference_fps = round((processed_frames / processing_time) if processing_time > 0 else 0.0, 4)

        inference_cfg = self.config.get("inference", {})
        model_name = Path(str(inference_cfg.get("model_path", "")).strip()).name
        if not model_name:
            model_name = "auto-selected-model.pt"

        model_path = str(inference_cfg.get("model_path", "")).strip()
        if not model_path:
            model_path = "auto-selected-latest-model"

        imgsz = int(inference_cfg.get("imgsz", 640))
        conf = float(inference_cfg.get("conf", 0.25))
        iou = float(inference_cfg.get("iou", 0.6))
        device = str(inference_cfg.get("device", "")).strip() or "auto"

        if successful_videos == total_videos:
            status = "success"
        elif successful_videos == 0:
            status = "failed"
        else:
            status = "partial_failed"

        return InferenceSummary(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            processing_time=processing_time,
            model_name=model_name,
            model_path=model_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            total_videos=total_videos,
            successful_videos=successful_videos,
            failed_videos=failed_videos,
            total_frames=total_frames,
            processed_frames=processed_frames,
            total_detections=total_detections,
            inference_fps=inference_fps,
            videos=video_rows,
            status=status,
            output_folder=str(output_dir.resolve()),
        )

    def _save_summary(self, summary: InferenceSummary, output_dir: Path) -> None:
        """Saves summary.json into inference run folder.

        Args:
            summary (InferenceSummary): Built summary object.
            output_dir (Path): Inference run output folder.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary.to_dict(), file, indent=2)

        logger.info("Saved inference summary to: %s", summary_path)

    def _save_config_snapshot(self, output_dir: Path) -> None:
        """Saves config_snapshot.yaml into inference run folder.

        Args:
            output_dir (Path): Inference run output folder.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = output_dir / "config_snapshot.yaml"

        with open(snapshot_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False, allow_unicode=True)

        logger.info("Saved inference config snapshot to: %s", snapshot_path)

    def _log_summary(self, summary: InferenceSummary) -> None:
        """Logs formatted inference summary to terminal.

        Args:
            summary (InferenceSummary): Built summary object.
        """
        logger.info("=" * 41)
        logger.info("INFERENCE SUMMARY")
        logger.info("=" * 41)
        logger.info("Model              : %s", summary.model_name)
        logger.info("Videos Processed   : %s", summary.total_videos)
        logger.info("Successful Videos  : %s", summary.successful_videos)
        logger.info("Failed Videos      : %s", summary.failed_videos)
        logger.info("Total Frames       : %s", summary.total_frames)
        logger.info("Total Detections   : %s", summary.total_detections)
        logger.info("Processing Time    : %s", self._format_processing_time(summary.processing_time))
        logger.info("Inference Speed    : %.1f FPS", summary.inference_fps)
        logger.info("-" * 41)

        for video in summary.videos:
            logger.info("Video              : %s", Path(video.input_video).name)
            logger.info("Frames             : %s", video.total_frames)
            logger.info("Processed Frames   : %s", video.processed_frames)
            logger.info("Detections         : %s", video.total_detections)
            logger.info("Processing Time    : %s", self._format_processing_time(video.processing_time))
            logger.info("Inference Speed    : %.1f FPS", video.inference_fps)
            logger.info("Status             : %s", video.status.upper())
            logger.info("Output             : %s", video.output_video)
            logger.info("=" * 41)

        logger.info("Run ID             : %s", summary.run_id)
        logger.info("Output Folder      : %s", summary.output_folder)
        logger.info("=" * 41)

    def _format_processing_time(self, seconds: float) -> str:
        """Formats seconds into compact minute-second representation.

        Args:
            seconds (float): Duration in seconds.

        Returns:
            str: Duration string in '<m>m <s>s' format.
        """
        total_seconds = max(int(round(seconds)), 0)
        minutes = total_seconds // 60
        remaining_seconds = total_seconds % 60
        return f"{minutes}m {remaining_seconds}s"

    def _resolve_output_root(self) -> Path:
        """Resolves inference output root directory from config.

        Returns:
            Path: Absolute output root path.
        """
        inference_cfg = self.config.get("inference", {})
        output_dir_cfg = str(
            inference_cfg.get("output_dir", "experiments/inference")
        ).strip()

        framework_root = Path(__file__).resolve().parent.parent
        output_root = Path(output_dir_cfg).expanduser()
        if not output_root.is_absolute():
            output_root = framework_root / output_root

        return output_root.resolve()

    def _load_configuration(self) -> Dict[str, Any]:
        """Loads framework configuration from configured source.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.

        Raises:
            RuntimeError: If configuration cannot be loaded.
        """
        try:
            config = self.config_loader.load()
            logger.info("Configuration loaded successfully.")
            return config
        except Exception as exc:
            message = f"Failed to load configuration: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc
