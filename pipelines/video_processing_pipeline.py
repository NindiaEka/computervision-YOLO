from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from services.frame_extraction_service import ExtractionResult, FrameExtractionService
from services.video_service import VideoInfo, VideoService
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class VideoProcessingVideoResult:
    """Represents per-video summary output for a processing run.

    Attributes:
        filename (str): Video file name with extension.
        fps (float): Source video FPS.
        duration_seconds (float): Source video duration in seconds.
        resolution (str): Source video resolution in WIDTHxHEIGHT format.
        frames_extracted (int): Total extracted frame images.
        output_dir (str): Output directory for extracted frames.
    """

    filename: str
    fps: float
    duration_seconds: float
    resolution: str
    frames_extracted: int
    output_dir: str


@dataclass(frozen=True)
class VideoProcessingSummary:
    """Represents the final summary produced by video processing pipeline.

    Attributes:
        run_id (str): Unique run identifier.
        started_at (datetime): Processing start timestamp.
        finished_at (datetime): Processing finish timestamp.
        processing_time (float): End-to-end elapsed seconds.
        mode (str): Extraction mode used for this run.
        total_videos (int): Total processed videos.
        total_frames (int): Total extracted frames across all videos.
        videos (List[VideoProcessingVideoResult]): Per-video summary rows.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    processing_time: float
    mode: str
    total_videos: int
    total_frames: int
    videos: List[VideoProcessingVideoResult]

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
            "mode": self.mode,
            "total_videos": self.total_videos,
            "total_frames": self.total_frames,
            "videos": [
                {
                    "filename": item.filename,
                    "fps": item.fps,
                    "duration_seconds": item.duration_seconds,
                    "resolution": item.resolution,
                    "frames_extracted": item.frames_extracted,
                    "output_dir": item.output_dir,
                }
                for item in self.videos
            ],
        }


class VideoProcessingPipeline:
    """Orchestrates end-to-end video processing workflow.

    This pipeline manages process ordering only. It does not include video
    validation or frame extraction implementation details.

    Attributes:
        config_loader (ConfigLoader): Configuration loader dependency.
        config (Dict[str, Any]): Loaded framework configuration.
        video_service (VideoService): Service responsible for video discovery.
        frame_extraction_service (FrameExtractionService): Service responsible for
            frame extraction.
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        video_service: Optional[VideoService] = None,
        frame_extraction_service: Optional[FrameExtractionService] = None,
    ) -> None:
        """Initializes pipeline dependencies using dependency injection.

        Args:
            config_loader (Optional[ConfigLoader]): Config loader instance.
            video_service (Optional[VideoService]): Video service instance.
            frame_extraction_service (Optional[FrameExtractionService]): Frame
                extraction service instance.
        """
        self.config_loader = config_loader or ConfigLoader()
        self.config = self._load_configuration()

        self.video_service = video_service or VideoService(
            config_loader=self.config_loader
        )
        self.frame_extraction_service = frame_extraction_service or FrameExtractionService(
            config_loader=self.config_loader
        )

    def run(self) -> VideoProcessingSummary:
        """Runs complete video processing workflow and returns summary object.

        Workflow:
            1. Validate injected dependencies.
            2. Prepare run metadata.
            3. Load videos from VideoService.
            4. Extract frames via FrameExtractionService.
            5. Build summary object.
            6. Save summary.json.
            7. Save config_snapshot.yaml.
            8. Log human-readable summary.

        Returns:
            VideoProcessingSummary: Final processing summary object.
        """
        self._validate_dependencies()

        run_id, started_at, mode, run_output_dir = self._prepare_run()
        videos = self._load_videos()
        extraction_results = self._extract_frames(videos=videos, run_id=run_id)

        summary = self._build_summary(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(),
            mode=mode,
            results=extraction_results,
        )

        self._save_summary(summary=summary, run_output_dir=run_output_dir)
        self._save_config_snapshot(run_output_dir=run_output_dir)
        self._log_summary(summary)

        return summary

    def _validate_dependencies(self) -> None:
        """Validates required pipeline dependencies.

        Raises:
            RuntimeError: If any required dependency is missing or invalid.
        """
        if self.config_loader is None:
            raise RuntimeError("ConfigLoader dependency is not available.")

        if self.video_service is None:
            raise RuntimeError("VideoService dependency is not available.")

        if self.frame_extraction_service is None:
            raise RuntimeError("FrameExtractionService dependency is not available.")

        if not hasattr(self.video_service, "get_videos"):
            raise RuntimeError("VideoService dependency must expose get_videos().")

        if not hasattr(self.frame_extraction_service, "extract_frames"):
            raise RuntimeError(
                "FrameExtractionService dependency must expose extract_frames()."
            )

    def _prepare_run(self) -> Tuple[str, datetime, str, Path]:
        """Prepares run metadata and output directory path.

        Returns:
            Tuple[str, datetime, str, Path]: Run ID, started timestamp,
            extraction mode, and run output directory.
        """
        started_at = datetime.now()
        run_id = started_at.strftime("run_%Y%m%d_%H%M%S")

        video_config = self.config.get("video", {})
        mode = str(video_config.get("mode", "fps")).strip().lower()
        output_dir_cfg = str(video_config.get("output_dir", "data/frames")).strip()

        framework_root = Path(__file__).resolve().parent.parent
        output_root = Path(output_dir_cfg).expanduser()
        if not output_root.is_absolute():
            output_root = framework_root / output_root

        run_output_dir = output_root.resolve() / run_id
        return run_id, started_at, mode, run_output_dir

    def _load_videos(self) -> List[VideoInfo]:
        """Loads videos to process using VideoService.

        Returns:
            List[VideoInfo]: Validated videos ready for extraction.
        """
        return self.video_service.get_videos()

    def _extract_frames(
        self,
        videos: List[VideoInfo],
        run_id: str,
    ) -> List[ExtractionResult]:
        """Extracts frames for loaded videos using FrameExtractionService.

        Args:
            videos (List[VideoInfo]): Videos to process.
            run_id (str): Current run identifier.

        Returns:
            List[ExtractionResult]: Structured extraction results.
        """
        return self.frame_extraction_service.extract_frames(videos=videos, run_id=run_id)

    def _build_summary(
        self,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        mode: str,
        results: List[ExtractionResult],
    ) -> VideoProcessingSummary:
        """Builds typed summary object from extraction results.

        Args:
            run_id (str): Run identifier.
            started_at (datetime): Pipeline start timestamp.
            finished_at (datetime): Pipeline finish timestamp.
            mode (str): Extraction mode.
            results (List[ExtractionResult]): Extraction result rows.

        Returns:
            VideoProcessingSummary: Structured summary object.
        """
        videos_summary = [
            VideoProcessingVideoResult(
                filename=result.video_info.filename,
                fps=result.video_info.fps,
                duration_seconds=result.video_info.duration_seconds,
                resolution=f"{result.video_info.width}x{result.video_info.height}",
                frames_extracted=result.frames_extracted,
                output_dir=str(result.output_dir),
            )
            for result in results
        ]

        total_frames = sum(item.frames_extracted for item in videos_summary)
        processing_time = (finished_at - started_at).total_seconds()

        return VideoProcessingSummary(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            processing_time=round(processing_time, 4),
            mode=mode,
            total_videos=len(videos_summary),
            total_frames=total_frames,
            videos=videos_summary,
        )

    def _save_summary(self, summary: VideoProcessingSummary, run_output_dir: Path) -> None:
        """Saves summary JSON file under run output directory.

        Args:
            summary (VideoProcessingSummary): Built summary object.
            run_output_dir (Path): Target run output directory.
        """
        run_output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_output_dir / "summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary.to_dict(), file, indent=2)

        logger.info("Saved video processing summary to: %s", summary_path)

    def _save_config_snapshot(self, run_output_dir: Path) -> None:
        """Saves configuration snapshot used for this run.

        Args:
            run_output_dir (Path): Target run output directory.
        """
        run_output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = run_output_dir / "config_snapshot.yaml"

        with open(snapshot_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False, allow_unicode=True)

        logger.info("Saved config snapshot to: %s", snapshot_path)

    def _log_summary(self, summary: VideoProcessingSummary) -> None:
        """Logs human-readable summary for terminal output.

        Args:
            summary (VideoProcessingSummary): Built summary object.
        """
        logger.info("\n" + "=" * 50)
        logger.info(" VIDEO PROCESSING SUMMARY ".center(50))
        logger.info("=" * 50)
        logger.info(f"Run ID          : {summary.run_id}")
        logger.info(f"Started At      : {summary.started_at.isoformat()}")
        logger.info(f"Finished At     : {summary.finished_at.isoformat()}")
        logger.info(f"Processing Time : {summary.processing_time} s")
        logger.info(f"Extraction Mode : {summary.mode}")
        logger.info(f"Total Videos    : {summary.total_videos}")
        logger.info(f"Total Frames    : {summary.total_frames}")
        logger.info("-" * 50)
        logger.info("Per Video:")

        for item in summary.videos:
            logger.info(f"Video Name      : {item.filename}")
            logger.info(f"Duration        : {item.duration_seconds} s")
            logger.info(f"FPS             : {item.fps}")
            logger.info(f"Resolution      : {item.resolution}")
            logger.info(f"Frames Extracted: {item.frames_extracted}")
            logger.info(f"Output Folder   : {item.output_dir}")
            logger.info("-" * 50)

        logger.info("=" * 50)

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
