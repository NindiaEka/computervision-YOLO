from __future__ import annotations

from datetime import datetime
import shutil
from pathlib import Path
from typing import Optional

from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


class ExportService:
    """Service to manage trained model export artifacts.

    This service copies the best trained model file from experiment output
    into a centralized trained models directory.

    Attributes:
        config_loader (ConfigLoader): Configuration loader dependency.
        trained_models_dir (Path): Destination directory for exported models.
        project_name (str): Project name used as model filename base.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes ExportService with configuration dependencies.

        Args:
            config_loader (Optional[ConfigLoader]): Configuration loader instance.
                If None, a default ConfigLoader will be used.
        """
        if config_loader is None:
            config_loader = ConfigLoader()

        self.config_loader = config_loader
        config = self.config_loader.get_config()

        output_config = config.get("output", {})
        project_config = config.get("project", {})

        self.trained_models_dir = Path(
            output_config.get("trained_models_dir", "trained_models")
        )
        self.project_name = str(project_config.get("name", "yolo11_model")).strip()

    def copy_best_model(self, best_model_path: str) -> str:
        """Copies best model to trained_models directory.

        The destination filename is derived from project name. If a file with
        the same name already exists, a timestamp suffix is appended to avoid
        overwriting existing model artifacts.

        Args:
            best_model_path (str): Source path of best.pt produced by training.

        Returns:
            str: Destination path of copied model file.

        Raises:
            FileNotFoundError: If source model file does not exist.
            RuntimeError: If copying fails for any reason.
        """
        source_path = Path(best_model_path)
        if not source_path.is_file():
            message = f"best.pt not found at source path: {source_path}"
            logger.error(message)
            raise FileNotFoundError(message)

        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

        destination_path = self._build_destination_path()

        try:
            shutil.copy2(source_path, destination_path)
            logger.info(
                f"Best model copied successfully from '{source_path}' "
                f"to '{destination_path}'"
            )
            return str(destination_path)
        except Exception as exc:
            message = f"Failed to copy best model: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

    def _build_destination_path(self) -> Path:
        """Builds destination file path for exported model.

        Returns:
            Path: Destination path in trained_models directory.
        """
        sanitized_project_name = self._sanitize_project_name(self.project_name)
        filename = f"{sanitized_project_name}.pt"
        destination_path = self.trained_models_dir / filename

        if destination_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{sanitized_project_name}_{timestamp}.pt"
            destination_path = self.trained_models_dir / filename

        return destination_path

    @staticmethod
    def _sanitize_project_name(project_name: str) -> str:
        """Converts project name into filesystem-friendly filename.

        Args:
            project_name (str): Raw project name from configuration.

        Returns:
            str: Sanitized filename-safe project name.
        """
        cleaned = "_".join(project_name.split())
        safe = "".join(ch for ch in cleaned if ch.isalnum() or ch in {"_", "-"})
        return safe or "yolo11_model"
