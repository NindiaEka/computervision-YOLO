from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from services.dataset_service import DatasetService
from services.export_service import ExportService
from services.roboflow_service import RoboflowService
from services.training_service import TrainingService
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


class TrainingPipeline:
    """Pipeline orchestrator for end-to-end YOLO11 training workflow.

    This class coordinates all steps in the training process using dependency
    injection. It does not contain service-specific business logic and only
    communicates through each service public API.

    Attributes:
        config_loader (ConfigLoader): Service used to load framework configuration.
        roboflow_service (RoboflowService): Service responsible for dataset retrieval.
        training_service (TrainingService): Service responsible for train/validation.
        export_service (ExportService): Service responsible for model artifact export.
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        roboflow_service: Optional[RoboflowService] = None,
        training_service: Optional[TrainingService] = None,
        export_service: Optional[ExportService] = None,
    ) -> None:
        """Initializes the training pipeline with injected dependencies.

        Args:
            config_loader (Optional[ConfigLoader]): Config loader instance.
            roboflow_service (Optional[RoboflowService]): Roboflow dataset service.
            training_service (Optional[TrainingService]): Model training service.
            export_service (Optional[ExportService]): Model export service.
        """
        self.config_loader = config_loader or ConfigLoader()
        self.config = self._load_configuration()
        self.roboflow_service = roboflow_service or RoboflowService(
            config_loader=self.config_loader
        )
        self.training_service = training_service or TrainingService(
            config_loader=self.config_loader
        )
        self.export_service = export_service or ExportService(
            config_loader=self.config_loader
        )

    def run(self) -> Dict[str, Any]:
        """Executes the full training flow.

        Flow:
            1. Load configuration.
            2. Download dataset from Roboflow if not available locally.
            3. Validate dataset using DatasetService.
            4. Show dataset summary.
            5. Run training and validation using TrainingService.
            6. Export best model into trained_models using ExportService.
            7. Show best/exported model locations.
            8. Show process summary.

        Returns:
            Dict[str, Any]: A structured summary of pipeline execution.

        Raises:
            RuntimeError: If any mandatory stage fails.
        """
        start_time = datetime.now()
        logger.info("Starting training pipeline...")

        data_yaml_path = self._prepare_dataset()
        dataset_summary = self._validate_and_summarize_dataset(data_yaml_path)
        best_model_path = self._run_training(data_yaml_path)
        exported_model_path = self._export_best_model(best_model_path)

        result = self._build_process_summary(
            start_time=start_time,
            config=self.config,
            data_yaml_path=data_yaml_path,
            dataset_summary=dataset_summary,
            best_model_path=best_model_path,
            exported_model_path=exported_model_path,
        )

        self._log_process_summary(result)
        return result

    def _load_configuration(self) -> Dict[str, Any]:
        """Loads framework configuration from config file.

        Returns:
            Dict[str, Any]: Configuration dictionary.

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

    def _prepare_dataset(self) -> str:
        """Ensures dataset is available and returns data.yaml path.

        Returns:
            str: Path to dataset data.yaml.

        Raises:
            RuntimeError: If dataset retrieval fails.
        """
        try:
            data_yaml_path = self.roboflow_service.get_dataset()
            logger.info("Dataset is ready for validation.")
            return data_yaml_path
        except Exception as exc:
            message = f"Failed to prepare dataset: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

    def _validate_and_summarize_dataset(self, data_yaml_path: str) -> Dict[str, Any]:
        """Validates dataset and generates summary via DatasetService.

        Args:
            data_yaml_path (str): Path to dataset YAML file.

        Returns:
            Dict[str, Any]: Dataset summary information.

        Raises:
            RuntimeError: If dataset is invalid or summary cannot be generated.
        """
        dataset_service = DatasetService(data_yaml_path)

        if not dataset_service.validate():
            message = "Dataset validation failed. Training process is stopped."
            logger.error(message)
            raise RuntimeError(message)

        logger.info("Dataset validation passed.")

        try:
            summary = dataset_service.summary()
            logger.info("Dataset summary generated.")
            return summary
        except Exception as exc:
            message = f"Failed to generate dataset summary: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

    def _run_training(self, data_yaml_path: str) -> str:
        """Runs training and validation process.

        Args:
            data_yaml_path (str): Path to dataset YAML file.

        Returns:
            str: Path to best model file.

        Raises:
            RuntimeError: If training stage fails.
        """
        try:
            best_model_path = self.training_service.run_training(data_yaml_path)
            logger.info(f"Best model saved at: {best_model_path}")
            return best_model_path
        except Exception as exc:
            message = f"Training stage failed: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

    def _export_best_model(self, best_model_path: str) -> str:
        """Exports best model artifact to trained_models directory.

        Args:
            best_model_path (str): Source path to best model from training.

        Returns:
            str: Destination path of exported model artifact.

        Raises:
            RuntimeError: If export stage fails.
        """
        try:
            exported_model_path = self.export_service.copy_best_model(best_model_path)
            logger.info(f"Exported model saved at: {exported_model_path}")
            return exported_model_path
        except Exception as exc:
            message = f"Export stage failed: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

    def _build_process_summary(
        self,
        start_time: datetime,
        config: Dict[str, Any],
        data_yaml_path: str,
        dataset_summary: Dict[str, Any],
        best_model_path: str,
        exported_model_path: str,
    ) -> Dict[str, Any]:
        """Builds final process summary payload.

        Args:
            start_time (datetime): Timestamp when pipeline started.
            config (Dict[str, Any]): Loaded framework configuration.
            data_yaml_path (str): Path to data.yaml.
            dataset_summary (Dict[str, Any]): Dataset summary payload.
            best_model_path (str): Best model path.
            exported_model_path (str): Exported model path.

        Returns:
            Dict[str, Any]: Consolidated pipeline summary.
        """
        finished_at = datetime.now()
        duration_seconds = (finished_at - start_time).total_seconds()

        project_name = config.get("project", {}).get("name", "unknown")

        return {
            "status": "success",
            "project": project_name,
            "started_at": start_time.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "data_yaml_path": data_yaml_path,
            "dataset_summary": dataset_summary,
            "best_model_path": best_model_path,
            "exported_model_path": exported_model_path,
        }

    def _log_process_summary(self, summary: Dict[str, Any]) -> None:
        """Logs pipeline execution summary.

        Args:
            summary (Dict[str, Any]): Final process summary.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PROCESS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status          : {summary['status']}")
        logger.info(f"Project         : {summary['project']}")
        logger.info(f"Data YAML       : {summary['data_yaml_path']}")
        logger.info(f"Best Model      : {summary['best_model_path']}")
        logger.info(f"Exported Model  : {summary['exported_model_path']}")
        logger.info(f"Duration (sec)  : {summary['duration_seconds']}")
        logger.info("=" * 60)
