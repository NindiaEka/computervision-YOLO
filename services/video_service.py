from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class VideoInfo:
    """Represents validated video metadata for downstream processing.

    Attributes:
        video_path (Path): Absolute path to the video file.
        filename (str): Video file name with extension.
        fps (float): Frames per second reported by the decoder.
        duration_seconds (float): Video duration in seconds.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        frame_count (int): Total number of frames in the video.
        file_size_bytes (int): Video file size in bytes.
    """

    video_path: Path
    filename: str
    fps: float
    duration_seconds: float
    width: int
    height: int
    frame_count: int
    file_size_bytes: int


class VideoService:
    """Service to prepare video inputs before extraction pipeline runs.

    This service reads configured input directory, validates supported video
    files, extracts metadata with OpenCV, and returns typed `VideoInfo`
    objects for downstream consumers.

    Attributes:
        config (Dict[str, Any]): Loaded framework configuration.
        video_config (Dict[str, Any]): Video section from configuration.
        input_dir (Path): Resolved absolute path to video input directory.
    """

    SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes VideoService and resolves configured input directory.

        Args:
            config_loader (Optional[ConfigLoader]): Configuration loader instance.
                If None, a default loader is created.

        Raises:
            RuntimeError: If configuration cannot be loaded.
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        try:
            self.config = config_loader.get_config()
        except RuntimeError:
            self.config = config_loader.load()

        self.video_config = self.config.get("video", {})
        input_dir_cfg = str(self.video_config.get("input_dir", "data/videos")).strip()

        framework_root = Path(__file__).resolve().parent.parent
        input_path = Path(input_dir_cfg).expanduser()
        if not input_path.is_absolute():
            input_path = framework_root / input_path

        self.input_dir = input_path.resolve()

    def get_videos(self) -> List[VideoInfo]:
        """Loads validated videos and returns their metadata.

        Returns:
            List[VideoInfo]: List of validated video metadata objects.

        Raises:
            FileNotFoundError: If configured input directory does not exist.
            NotADirectoryError: If configured input path is not a directory.
            RuntimeError: If no supported videos are found.
        """
        self._validate_input_directory()

        video_files = self._list_supported_video_files()
        if not video_files:
            message = (
                "No supported video files found in input directory: "
                f"{self.input_dir}"
            )
            logger.error(message)
            raise RuntimeError(message)

        videos: List[VideoInfo] = []
        for video_path in video_files:
            videos.append(self._extract_video_info(video_path))

        logger.info(
            "Loaded %d valid video(s) from '%s'.",
            len(videos),
            self.input_dir,
        )
        return videos

    def _validate_input_directory(self) -> None:
        """Validates that configured input directory exists and is a folder.

        Raises:
            FileNotFoundError: If input directory does not exist.
            NotADirectoryError: If input path is not a directory.
        """
        if not self.input_dir.exists():
            message = f"Video input directory not found: {self.input_dir}"
            logger.error(message)
            raise FileNotFoundError(message)

        if not self.input_dir.is_dir():
            message = f"Configured video input path is not a directory: {self.input_dir}"
            logger.error(message)
            raise NotADirectoryError(message)

    def _list_supported_video_files(self) -> List[Path]:
        """Lists supported video files from configured input directory.

        Returns:
            List[Path]: Sorted list of supported video file paths.
        """
        video_files = [
            path
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        unsupported_files = [
            path.name
            for path in self.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() not in self.SUPPORTED_EXTENSIONS
        ]

        if unsupported_files:
            logger.warning(
                "Ignoring %d unsupported file(s): %s",
                len(unsupported_files),
                ", ".join(sorted(unsupported_files)),
            )

        return sorted(video_files, key=lambda p: p.name.lower())

    def _extract_video_info(self, video_path: Path) -> VideoInfo:
        """Extracts metadata from a video file using OpenCV.

        Args:
            video_path (Path): Absolute path to the video file.

        Returns:
            VideoInfo: Extracted metadata for the given video.

        Raises:
            RuntimeError: If OpenCV fails to open the video file.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            message = f"Failed to open video file with OpenCV: {video_path}"
            logger.error(message)
            raise RuntimeError(message)

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

        duration_seconds = float(frame_count / fps) if fps > 0 else 0.0
        file_size_bytes = video_path.stat().st_size

        video_info = VideoInfo(
            video_path=video_path.resolve(),
            filename=video_path.name,
            fps=round(fps, 4),
            duration_seconds=round(duration_seconds, 4),
            width=width,
            height=height,
            frame_count=frame_count,
            file_size_bytes=file_size_bytes,
        )

        logger.info(
            "Video metadata loaded: %s | fps=%.4f | duration=%.4fs | "
            "resolution=%dx%d | frames=%d | size=%d bytes",
            video_info.filename,
            video_info.fps,
            video_info.duration_seconds,
            video_info.width,
            video_info.height,
            video_info.frame_count,
            video_info.file_size_bytes,
        )

        return video_info
