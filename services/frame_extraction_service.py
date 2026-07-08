from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from services.video_service import VideoInfo
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class ExtractionResult:
    """Represents extraction outcome for a single video.

    Attributes:
        run_id (str): Unique run identifier for the extraction batch.
        video_info (VideoInfo): Source video metadata.
        frames_extracted (int): Total frame files generated.
        output_dir (Path): Directory where frames were written.
        processing_time (float): Processing duration in seconds.
    """

    run_id: str
    video_info: VideoInfo
    frames_extracted: int
    output_dir: Path
    processing_time: float


class FrameExtractionService:
    """Service to extract image frames from validated videos using FFmpeg.

    This service consumes video metadata objects from VideoService and produces
    structured extraction results for downstream summary generation.

    Attributes:
        config (Dict[str, Any]): Loaded framework configuration.
        video_config (Dict[str, Any]): Video configuration section.
        output_root (Path): Base directory for extracted frame outputs.
        mode (str): Extraction mode, either fps or interval.
        fps (float): Target frames per second when mode is fps.
        interval_seconds (float): Capture interval in seconds when mode is interval.
        image_format (str): Output image format/extension.
        ffmpeg_path (str): FFmpeg executable command or absolute path.
    """

    SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes FrameExtractionService using loaded configuration.

        Args:
            config_loader (Optional[ConfigLoader]): Configuration loader instance.
                If None, a default loader is created.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        try:
            self.config = config_loader.get_config()
        except RuntimeError:
            self.config = config_loader.load()

        self.video_config = self.config.get("video", {})
        extraction_cfg = self.video_config.get("extraction", {})

        framework_root = Path(__file__).resolve().parent.parent
        output_dir_cfg = str(self.video_config.get("output_dir", "data/frames")).strip()
        output_path = Path(output_dir_cfg).expanduser()
        if not output_path.is_absolute():
            output_path = framework_root / output_path
        self.output_root = output_path.resolve()

        self.mode = str(
            extraction_cfg.get("mode", self.video_config.get("mode", "fps"))
        ).strip().lower()
        self.fps = float(extraction_cfg.get("fps", self.video_config.get("fps", 1)))
        self.interval_seconds = float(
            extraction_cfg.get(
                "interval_seconds",
                self.video_config.get("interval_seconds", 5),
            )
        )

        image_format = str(self.video_config.get("image_format", "jpg")).strip().lower()
        self.image_format = image_format.lstrip(".")
        self.ffmpeg_path = str(self.video_config.get("ffmpeg_path", "ffmpeg")).strip()

        self._validate_configuration()

    def extract_frames(
        self,
        videos: List[VideoInfo],
        run_id: Optional[str] = None,
    ) -> List[ExtractionResult]:
        """Extracts frames from provided videos and returns structured results.

        Args:
            videos (List[VideoInfo]): Videos to process.
            run_id (Optional[str]): Optional external run ID. If omitted, an ID
                with format run_YYYYMMDD_HHMMSS is generated.

        Returns:
            List[ExtractionResult]: Per-video extraction outcomes.

        Raises:
            RuntimeError: If frame extraction fails for any video.
            ValueError: If videos list is empty.
        """
        if not videos:
            raise ValueError("No videos provided for frame extraction.")

        self._validate_ffmpeg_available()

        resolved_run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_output_dir = self._resolve_unique_run_output_dir(resolved_run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        results: List[ExtractionResult] = []
        for video_info in videos:
            result = self._extract_single_video(
                video_info=video_info,
                run_id=resolved_run_id,
                run_output_dir=run_output_dir,
            )
            results.append(result)

        logger.info(
            "Frame extraction completed for %d video(s) under run '%s'.",
            len(results),
            run_output_dir.name,
        )
        return results

    def _resolve_unique_run_output_dir(self, run_id: str) -> Path:
        """Builds a unique run output directory without deleting old runs.

        Args:
            run_id (str): Preferred run identifier.

        Returns:
            Path: Unique run output path under configured output root.
        """
        base_dir = self.output_root / run_id
        if not base_dir.exists():
            return base_dir

        counter = 1
        while True:
            candidate = self.output_root / f"{run_id}_{counter:03d}"
            if not candidate.exists():
                logger.warning(
                    "Run directory already exists. Using next run ID: %s",
                    candidate.name,
                )
                return candidate
            counter += 1

    def _validate_configuration(self) -> None:
        """Validates extraction mode and configurable extraction parameters.

        Raises:
            ValueError: If mode, rate, or image format configuration is invalid.
        """
        if self.mode not in {"fps", "interval"}:
            raise ValueError("video.extraction.mode must be either 'fps' or 'interval'.")

        if self.mode == "fps" and self.fps <= 0:
            raise ValueError(
                "video.extraction.fps must be greater than 0 when mode is 'fps'."
            )

        if self.mode == "interval" and self.interval_seconds <= 0:
            raise ValueError(
                "video.extraction.interval_seconds must be greater than 0 when mode "
                "is 'interval'."
            )

        if self.image_format not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                "video.image_format must be one of: "
                f"{sorted(self.SUPPORTED_IMAGE_FORMATS)}"
            )

    def _validate_ffmpeg_available(self) -> None:
        """Ensures configured FFmpeg executable is available and runnable.

        Raises:
            RuntimeError: If ffmpeg executable cannot be resolved or executed.
        """
        ffmpeg_candidate = self.ffmpeg_path
        candidate_path = Path(ffmpeg_candidate).expanduser()

        # Priority 1: command available in PATH.
        in_path = shutil.which(ffmpeg_candidate)

        # Priority 2: absolute file path provided explicitly.
        if candidate_path.is_absolute() and not candidate_path.is_file():
            raise RuntimeError(
                "Configured FFmpeg path is absolute but file does not exist: "
                f"{candidate_path}"
            )

        # Priority 3: runtime probe to handle aliases/wrappers not resolved by which.
        try:
            subprocess.run(
                [ffmpeg_candidate, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            source_hint = (
                "PATH" if in_path is not None else "alias-or-custom-command"
            )
            raise RuntimeError(
                "FFmpeg executable is not available or failed to run. "
                f"Configured ffmpeg_path='{ffmpeg_candidate}', source='{source_hint}'. "
                "Verify installation and config video.ffmpeg_path."
            ) from exc

    def _extract_single_video(
        self,
        video_info: VideoInfo,
        run_id: str,
        run_output_dir: Path,
    ) -> ExtractionResult:
        """Extracts frames from one video and returns extraction result.

        Args:
            video_info (VideoInfo): Source video metadata.
            run_id (str): Current extraction run identifier.
            run_output_dir (Path): Root output folder for the run.

        Returns:
            ExtractionResult: Structured extraction outcome.

        Raises:
            RuntimeError: If FFmpeg command fails.
        """
        video_output_dir = run_output_dir / video_info.video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        output_pattern = video_output_dir / f"frame_%06d.{self.image_format}"
        command = self._build_ffmpeg_command(
            video_path=video_info.video_path,
            output_pattern=output_pattern,
        )

        start = time.perf_counter()
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - start

        if process.returncode != 0:
            error_message = (
                f"FFmpeg extraction failed for '{video_info.filename}'. "
                f"stderr: {process.stderr.strip()}"
            )
            logger.error(error_message)
            raise RuntimeError(error_message)

        frames_extracted = self._count_extracted_frames(video_output_dir)

        result = ExtractionResult(
            run_id=run_id,
            video_info=video_info,
            frames_extracted=frames_extracted,
            output_dir=video_output_dir.resolve(),
            processing_time=round(elapsed, 4),
        )

        logger.info(
            "Extracted %d frame(s) from '%s' into '%s' in %.4f seconds.",
            result.frames_extracted,
            result.video_info.filename,
            result.output_dir,
            result.processing_time,
        )

        return result

    def _build_ffmpeg_command(self, video_path: Path, output_pattern: Path) -> List[str]:
        """Builds FFmpeg command based on configured extraction mode.

        Args:
            video_path (Path): Source video path.
            output_pattern (Path): Output frame naming pattern.

        Returns:
            List[str]: Tokenized FFmpeg command.
        """
        if self.mode == "fps":
            filter_expr = f"fps={self.fps}"
        else:
            filter_expr = f"fps=1/{self.interval_seconds}"

        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            filter_expr,
            str(output_pattern),
        ]

    def _count_extracted_frames(self, output_dir: Path) -> int:
        """Counts extracted frame files in an output directory.

        Args:
            output_dir (Path): Directory containing extracted frame files.

        Returns:
            int: Total extracted frame count.
        """
        extension = f".{self.image_format}"
        return sum(1 for file in output_dir.iterdir() if file.suffix.lower() == extension)
