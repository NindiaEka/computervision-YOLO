from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional

from ultralytics import YOLO

from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Represents validation output metrics and generated artifacts.

    Attributes:
        model_name (str): Model filename used for validation.
        model_path (str): Absolute model path.
        dataset (str): Absolute dataset YAML path.
        precision (float): Mean precision metric.
        recall (float): Mean recall metric.
        map50 (float): mAP at IoU 0.50.
        map50_95 (float): mAP at IoU 0.50:0.95.
        processing_time (float): Validation duration in seconds.
        artifacts (Dict[str, str]): Optional generated artifact file paths.
    """

    model_name: str
    model_path: str
    dataset: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    processing_time: float
    artifacts: Dict[str, str]


class ValidationService:
    """Service to execute standalone model validation using Ultralytics.

    This service validates an existing model (.pt) against an existing dataset
    YAML and returns typed validation metrics and optional artifact references.

    Attributes:
        config (Dict[str, Any]): Loaded framework configuration.
        validation_config (Dict[str, Any]): Validation-specific config section.
        output_config (Dict[str, Any]): Output directory configuration section.
        framework_root (Path): Root directory of the framework project.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes ValidationService with dependency-injected config loader.

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

        self.validation_config = self.config.get("validation", {})
        self.output_config = self.config.get("output", {})
        self.framework_root = Path(__file__).resolve().parent.parent

    def validate(self, run_id: Optional[str] = None) -> ValidationResult:
        """Runs model validation and returns structured result.

        Args:
            run_id (Optional[str]): Optional run identifier used as Ultralytics
                run name for validation output. If None, a default name is used.

        Returns:
            ValidationResult: Structured validation metrics and artifacts.

        Raises:
            FileNotFoundError: If model or dataset YAML cannot be resolved.
            RuntimeError: If validation execution fails.
        """
        started = time.perf_counter()

        model_path = self._resolve_model_path()
        dataset_yaml_path = self._resolve_dataset_yaml_path()

        logger.info("Starting validation using model: %s", model_path)
        logger.info("Using validation dataset: %s", dataset_yaml_path)

        try:
            model = YOLO(str(model_path))
            val_kwargs = self._build_val_kwargs(dataset_yaml_path, run_id)
            results = model.val(**val_kwargs)
        except Exception as exc:
            message = f"Validation process failed: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        precision, recall, map50, map50_95 = self._extract_metrics(results)
        artifacts = self._collect_optional_artifacts(results)

        processing_time = round(time.perf_counter() - started, 4)

        result = ValidationResult(
            model_name=model_path.name,
            model_path=str(model_path),
            dataset=str(dataset_yaml_path),
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            processing_time=processing_time,
            artifacts=artifacts,
        )

        logger.info(
            "Validation completed: precision=%.6f recall=%.6f map50=%.6f map50-95=%.6f",
            result.precision,
            result.recall,
            result.map50,
            result.map50_95,
        )

        return result

    def _resolve_model_path(self) -> Path:
        """Resolves validation model path from config.

        Resolution priority:
            1. validation.model_path
            2. latest .pt under output.trained_models_dir

        Returns:
            Path: Absolute model file path.

        Raises:
            FileNotFoundError: If no valid model path can be found.
        """
        configured_model_path = str(
            self.validation_config.get("model_path", "")
        ).strip()

        if configured_model_path:
            model_path = self._resolve_path(configured_model_path)
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Configured validation model not found: {model_path}"
                )
            return model_path

        trained_models_dir_cfg = str(
            self.output_config.get("trained_models_dir", "trained_models")
        ).strip()
        trained_models_dir = self._resolve_path(trained_models_dir_cfg)

        if not trained_models_dir.is_dir():
            raise FileNotFoundError(
                "Validation model path is not configured and trained models "
                f"directory does not exist: {trained_models_dir}"
            )

        candidate_models = sorted(
            trained_models_dir.glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not candidate_models:
            raise FileNotFoundError(
                "Validation model path is not configured and no .pt model was "
                f"found in: {trained_models_dir}"
            )

        selected_model = candidate_models[0].resolve()
        logger.info("Using latest trained model for validation: %s", selected_model)
        return selected_model

    def _resolve_dataset_yaml_path(self) -> Path:
        """Resolves dataset YAML path for validation.

        Resolution priority:
            1. validation.dataset_yaml_path
            2. datasets/<roboflow_project>-<roboflow_version>/data.yaml
            3. first YAML file in datasets/<roboflow_project>-<roboflow_version>

        Returns:
            Path: Absolute dataset YAML path.

        Raises:
            FileNotFoundError: If dataset YAML cannot be resolved.
        """
        configured_yaml_path = str(
            self.validation_config.get("dataset_yaml_path", "")
        ).strip()

        if configured_yaml_path:
            yaml_path = self._resolve_path(configured_yaml_path)
            if not yaml_path.is_file():
                raise FileNotFoundError(
                    f"Configured validation dataset YAML not found: {yaml_path}"
                )
            return yaml_path

        datasets_dir_cfg = str(self.output_config.get("datasets_dir", "datasets")).strip()
        datasets_dir = self._resolve_path(datasets_dir_cfg)

        roboflow_cfg = self.config.get("roboflow", {})
        project_name = str(roboflow_cfg.get("project", "")).strip()
        version = roboflow_cfg.get("version", 1)

        dataset_dir = datasets_dir / f"{project_name}-{version}"
        default_yaml = dataset_dir / "data.yaml"

        if default_yaml.is_file():
            return default_yaml.resolve()

        if dataset_dir.is_dir():
            yaml_files = sorted(dataset_dir.glob("*.yaml"))
            if yaml_files:
                logger.warning(
                    "Default data.yaml not found. Using alternative dataset YAML: %s",
                    yaml_files[0],
                )
                return yaml_files[0].resolve()

        raise FileNotFoundError(
            "Validation dataset YAML is not configured and automatic dataset YAML "
            f"resolution failed under: {dataset_dir}"
        )

    def _build_val_kwargs(self, dataset_yaml_path: Path, run_id: Optional[str]) -> Dict[str, Any]:
        """Builds kwargs for Ultralytics YOLO validation.

        Args:
            dataset_yaml_path (Path): Dataset YAML used for validation.
            run_id (Optional[str]): Optional run name.

        Returns:
            Dict[str, Any]: Ultralytics model.val() keyword arguments.
        """
        output_root = self._resolve_path(
            str(self.validation_config.get("output_dir", "experiments/validation")).strip()
        )

        run_name = str(run_id or self.validation_config.get("run_name", "validation")).strip()
        if not run_name:
            run_name = "validation"

        return {
            "data": str(dataset_yaml_path),
            "split": str(self.validation_config.get("split", "val")).strip(),
            "conf": float(self.validation_config.get("conf", 0.25)),
            "iou": float(self.validation_config.get("iou", 0.6)),
            "imgsz": int(self.validation_config.get("imgsz", 640)),
            "batch": int(self.validation_config.get("batch", 16)),
            "device": str(self.validation_config.get("device", "")).strip(),
            "plots": bool(self.validation_config.get("plots", True)),
            "project": str(output_root),
            "name": run_name,
            "exist_ok": bool(self.validation_config.get("exist_ok", False)),
        }

    def _extract_metrics(self, results: Any) -> tuple[float, float, float, float]:
        """Extracts required validation metrics from Ultralytics results.

        Args:
            results (Any): Ultralytics validation result object.

        Returns:
            tuple[float, float, float, float]: Precision, recall, mAP50,
            and mAP50-95.
        """
        box_metrics = getattr(results, "box", None)

        precision = float(getattr(box_metrics, "mp", 0.0) if box_metrics else 0.0)
        recall = float(getattr(box_metrics, "mr", 0.0) if box_metrics else 0.0)
        map50 = float(getattr(box_metrics, "map50", 0.0) if box_metrics else 0.0)
        map50_95 = float(getattr(box_metrics, "map", 0.0) if box_metrics else 0.0)

        # Fallback for result schemas exposing metrics via results_dict.
        if precision == 0.0 and recall == 0.0 and map50 == 0.0 and map50_95 == 0.0:
            result_dict = getattr(results, "results_dict", {}) or {}
            precision = float(result_dict.get("metrics/precision(B)", precision))
            recall = float(result_dict.get("metrics/recall(B)", recall))
            map50 = float(result_dict.get("metrics/mAP50(B)", map50))
            map50_95 = float(result_dict.get("metrics/mAP50-95(B)", map50_95))

        return (
            round(precision, 6),
            round(recall, 6),
            round(map50, 6),
            round(map50_95, 6),
        )

    def _collect_optional_artifacts(self, results: Any) -> Dict[str, str]:
        """Collects optional validation artifacts if generated by Ultralytics.

        Args:
            results (Any): Ultralytics validation result object.

        Returns:
            Dict[str, str]: Artifact key to absolute path mapping.
        """
        save_dir_value = getattr(results, "save_dir", None)
        if not save_dir_value:
            return {}

        save_dir = Path(str(save_dir_value)).resolve()
        if not save_dir.is_dir():
            return {}

        artifacts: Dict[str, str] = {}
        artifact_patterns = {
            "confusion_matrix": "confusion_matrix*.png",
            "precision_recall_curve": "PR_curve*.png",
            "f1_curve": "F1_curve*.png",
        }

        for artifact_name, pattern in artifact_patterns.items():
            matches = sorted(save_dir.glob(pattern))
            if matches:
                artifacts[artifact_name] = str(matches[0].resolve())

        return artifacts

    def _resolve_path(self, path_value: str) -> Path:
        """Resolves path relative to framework root when path is not absolute.

        Args:
            path_value (str): Path string from configuration.

        Returns:
            Path: Resolved absolute path.
        """
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self.framework_root / path
        return path.resolve()
