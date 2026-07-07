from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from services.validation_service import ValidationResult, ValidationService
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class ValidationMetrics:
    """Represents required validation metrics.

    Attributes:
        precision (float): Mean precision metric.
        recall (float): Mean recall metric.
        map50 (float): mAP at IoU 0.50.
        map50_95 (float): mAP at IoU 0.50:0.95.
    """

    precision: float
    recall: float
    map50: float
    map50_95: float


@dataclass(frozen=True)
class ValidationArtifacts:
    """Represents optional validation artifact file paths.

    Attributes:
        confusion_matrix (Optional[str]): Path to confusion matrix image.
        pr_curve (Optional[str]): Path to precision-recall curve image.
        f1_curve (Optional[str]): Path to F1 curve image.
        results_png (Optional[str]): Path to results plot image.
        results_csv (Optional[str]): Path to tabular results CSV.
    """

    confusion_matrix: Optional[str]
    pr_curve: Optional[str]
    f1_curve: Optional[str]
    results_png: Optional[str]
    results_csv: Optional[str]


@dataclass(frozen=True)
class ValidationSummary:
    """Represents final validation pipeline summary output.

    Attributes:
        run_id (str): Validation run identifier.
        started_at (datetime): Pipeline start timestamp.
        finished_at (datetime): Pipeline finish timestamp.
        processing_time (float): End-to-end elapsed seconds.
        model_name (str): Model file name used in validation.
        model_path (str): Model file path used in validation.
        dataset (str): Dataset YAML path used in validation.
        metrics (ValidationMetrics): Nested validation metrics object.
        artifacts (ValidationArtifacts): Nested optional artifact paths object.
        status (str): Validation execution status.
        output_folder (str): Validation run output directory.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime
    processing_time: float
    model_name: str
    model_path: str
    dataset: str
    metrics: ValidationMetrics
    artifacts: ValidationArtifacts
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
            "dataset": self.dataset,
            "metrics": {
                "precision": self.metrics.precision,
                "recall": self.metrics.recall,
                "map50": self.metrics.map50,
                "map50_95": self.metrics.map50_95,
            },
            "artifacts": {
                "confusion_matrix": self.artifacts.confusion_matrix,
                "pr_curve": self.artifacts.pr_curve,
                "f1_curve": self.artifacts.f1_curve,
                "results_png": self.artifacts.results_png,
                "results_csv": self.artifacts.results_csv,
            },
            "status": self.status,
            "output_folder": self.output_folder,
        }


class ValidationPipeline:
    """Orchestrates standalone validation workflow.

    This pipeline only coordinates process order and persistence. Validation
    business logic remains in ValidationService.

    Attributes:
        config_loader (ConfigLoader): Configuration loader dependency.
        config (Dict[str, Any]): Loaded framework configuration.
        validation_service (ValidationService): Validation business service.
    """

    def __init__(
        self,
        config_loader: Optional[ConfigLoader] = None,
        validation_service: Optional[ValidationService] = None,
    ) -> None:
        """Initializes ValidationPipeline using dependency injection.

        Args:
            config_loader (Optional[ConfigLoader]): Config loader instance.
            validation_service (Optional[ValidationService]): Validation service
                instance.
        """
        self.config_loader = config_loader or ConfigLoader()
        self.config = self._load_configuration()
        self.validation_service = validation_service or ValidationService(
            config_loader=self.config_loader
        )

    def run(self) -> ValidationSummary:
        """Executes full validation pipeline and returns typed summary.

        Returns:
            ValidationSummary: Final validation summary.
        """
        self._validate_dependencies()

        run_id, started_at, output_dir = self._prepare_run()
        validation_result = self._run_validation(run_id)
        summary = self._build_summary(
            run_id=run_id,
            started_at=started_at,
            output_dir=output_dir,
            validation_result=validation_result,
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

        if self.validation_service is None:
            raise RuntimeError("ValidationService dependency is not available.")

        if not hasattr(self.validation_service, "validate"):
            raise RuntimeError("ValidationService dependency must expose validate().")

    def _prepare_run(self) -> Tuple[str, datetime, Path]:
        """Prepares run metadata and validation output directory.

        Returns:
            Tuple[str, datetime, Path]: Run ID, start timestamp, and output path.
        """
        started_at = datetime.now()
        run_id = started_at.strftime("run_%Y%m%d_%H%M%S")

        output_root = self._resolve_output_root()
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return run_id, started_at, output_dir

    def _run_validation(self, run_id: str) -> ValidationResult:
        """Runs validation process via ValidationService only.

        Args:
            run_id (str): Validation run identifier.

        Returns:
            ValidationResult: Service validation result.
        """
        return self.validation_service.validate(run_id=run_id)

    def _build_summary(
        self,
        run_id: str,
        started_at: datetime,
        output_dir: Path,
        validation_result: ValidationResult,
    ) -> ValidationSummary:
        """Builds validation summary object from service result.

        Args:
            run_id (str): Validation run identifier.
            started_at (datetime): Validation start timestamp.
            output_dir (Path): Validation output folder.
            validation_result (ValidationResult): Result from ValidationService.

        Returns:
            ValidationSummary: Structured validation summary.
        """
        finished_at = datetime.now()
        processing_time = round((finished_at - started_at).total_seconds(), 4)

        artifacts_map = validation_result.artifacts or {}
        artifacts = ValidationArtifacts(
            confusion_matrix=artifacts_map.get("confusion_matrix"),
            pr_curve=artifacts_map.get("precision_recall_curve")
            or artifacts_map.get("pr_curve"),
            f1_curve=artifacts_map.get("f1_curve"),
            results_png=artifacts_map.get("results_png"),
            results_csv=artifacts_map.get("results_csv"),
        )

        metrics = ValidationMetrics(
            precision=validation_result.precision,
            recall=validation_result.recall,
            map50=validation_result.map50,
            map50_95=validation_result.map50_95,
        )

        return ValidationSummary(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            processing_time=processing_time,
            model_name=validation_result.model_name,
            model_path=validation_result.model_path,
            dataset=validation_result.dataset,
            metrics=metrics,
            artifacts=artifacts,
            status="success",
            output_folder=str(output_dir.resolve()),
        )

    def _save_summary(self, summary: ValidationSummary, output_dir: Path) -> None:
        """Saves summary.json into validation run folder.

        Args:
            summary (ValidationSummary): Built summary object.
            output_dir (Path): Validation run output folder.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"

        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary.to_dict(), file, indent=2)

        logger.info("Saved validation summary to: %s", summary_path)

    def _save_config_snapshot(self, output_dir: Path) -> None:
        """Saves config_snapshot.yaml into validation run folder.

        Args:
            output_dir (Path): Validation run output folder.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = output_dir / "config_snapshot.yaml"

        with open(snapshot_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False, allow_unicode=True)

        logger.info("Saved validation config snapshot to: %s", snapshot_path)

    def _log_summary(self, summary: ValidationSummary) -> None:
        """Logs formatted validation summary to terminal.

        Args:
            summary (ValidationSummary): Built summary object.
        """
        logger.info("=" * 40)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 40)
        logger.info("Run ID          : %s", summary.run_id)
        logger.info("Model           : %s", summary.model_name)
        logger.info("Dataset         : %s", summary.dataset)
        logger.info("Precision       : %s", summary.metrics.precision)
        logger.info("Recall          : %s", summary.metrics.recall)
        logger.info("mAP50           : %s", summary.metrics.map50)
        logger.info("mAP50-95        : %s", summary.metrics.map50_95)
        logger.info("Processing Time : %s", summary.processing_time)
        logger.info("Output Folder   : %s", summary.output_folder)
        logger.info("=" * 40)

    def _resolve_output_root(self) -> Path:
        """Resolves validation output root directory from config.

        Returns:
            Path: Absolute output root path.
        """
        validation_cfg = self.config.get("validation", {})
        output_dir_cfg = str(
            validation_cfg.get("output_dir", "experiments/validation")
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
