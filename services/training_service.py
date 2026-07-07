from pathlib import Path
from typing import Dict, Any, Optional

import torch
from ultralytics import YOLO

from utils.config import ConfigLoader
from utils.logger import Logger

logger = Logger.get_logger(__name__)


class TrainingService:
    """Service to handle the training and validation of Ultralytics YOLO11 models.
    
    Attributes:
        config (Dict[str, Any]): The loaded configuration dictionary.
        model_config (Dict[str, Any]): Configuration specific to the YOLO model.
        train_config (Dict[str, Any]): Configuration for the training process.
        val_config (Dict[str, Any]): Configuration for the validation process.
        output_config (Dict[str, Any]): Configuration for output directories.
        project_name (str): The name of the project/experiment.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes the TrainingService and parses network configs.
        
        Args:
            config_loader (Optional[ConfigLoader]): ConfigLoader instance. 
                Initializes a new one if not provided.
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self.config = config_loader.get_config()
        
        self.model_config = self.config.get("model", {})
        self.train_config = self.config.get("training", {})
        self.val_config = self.config.get("validation", {})
        self.output_config = self.config.get("output", {})

        # Resolve paths from framework root, not process working directory,
        # to keep this project standalone even when invoked externally.
        self.framework_root = Path(__file__).resolve().parent.parent
        self.output_dir = self._resolve_output_dir()
        self.experiment_name = str(
            self.train_config.get(
                "experiment_name",
                self.config.get("project", {}).get("name", "YOLO11_Run"),
            )
        ).strip() or "YOLO11_Run"

    def run_training(self, data_yaml_path: str) -> str:
        """Executes the YOLO11 model training and subsequent validation.
        
        Args:
            data_yaml_path (str): The absolute or relative path to the dataset's data.yaml.
            
        Returns:
            str: Path to the best trained model (best.pt).
            
        Raises:
            RuntimeError: If training fails to execute properly.
        """
        device = self._select_device()
        model_path = self._get_pretrained_path()
        experiments_dir = str(self.output_dir)
        
        logger.info(f"Loading YOLO11 model from: {model_path}")
        try:
            model = YOLO(model_path)
        except Exception as e:
            error_msg = f"Failed to initialize YOLO model. Ensure path/name is correct. Error: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Starting training on device: {device}")
        
        # Prepare training arguments from config
        train_kwargs = {
            "data": data_yaml_path,
            "project": experiments_dir,
            "name": self.experiment_name,
            "epochs": self.train_config.get("epochs", 100),
            "batch": self.train_config.get("batch", 16),
            "imgsz": self.train_config.get("imgsz", 640),
            "device": device,
            "optimizer": self.train_config.get("optimizer", "auto"),
            "lr0": self.train_config.get("lr0", 0.01),
            "lrf": self.train_config.get("lrf", 0.01),
            "patience": self.train_config.get("patience", 50),
            "workers": self.train_config.get("workers", 8),
            "exist_ok": self.train_config.get("exist_ok", False),
        }
        
        logger.info("=" * 60)
        logger.info(f"Framework Root : {self.framework_root}")
        logger.info(f"Output Dir     : {self.output_dir}")
        logger.info(f"Experiment     : {self.experiment_name}")
        logger.info(f"YOLO Project   : {experiments_dir}")
        logger.info("=" * 60)
        try:
            # Training Phase
            model.train(**train_kwargs)
            logger.info("Training completed successfully.")
            
            # Validation Phase
            logger.info("Starting validation phase...")
            val_kwargs = {
                "conf": self.val_config.get("conf", 0.25),
                "iou": self.val_config.get("iou", 0.6),
                "half": self.val_config.get("half", True),
                "plots": self.val_config.get("plots", True),
                "device": device
            }
            model.val(**val_kwargs)
            logger.info("Validation completed successfully.")
            
        except Exception as e:
            error_msg = f"An error occurred during training or validation: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Retrieve the best model path
        best_model_path = self._get_best_model_path(model, experiments_dir)
        return str(best_model_path)

    def _resolve_output_dir(self) -> Path:
        """Resolves training output directory to an absolute framework-local path."""
        output_dir_cfg = str(
            self.train_config.get(
                "output_dir",
                self.output_config.get("experiments_dir", "experiments"),
            )
        ).strip()

        output_path = Path(output_dir_cfg).expanduser()
        if not output_path.is_absolute():
            output_path = self.framework_root / output_path

        return output_path.resolve()

    def _select_device(self) -> str:
        """Determines the appropriate device to run the model on.
        
        Checks the configuration first. If not specified, automatically assigns
        GPU if CUDA is available, otherwise defaults to CPU.
        
        Returns:
            str: The device specifier (e.g., '0', 'cpu', 'cuda:0').
        """
        device_cfg = str(self.train_config.get("device", "")).strip()
        
        if device_cfg:
            logger.info(f"Using configured device: '{device_cfg}'")
            return device_cfg
            
        if torch.cuda.is_available():
            logger.info("GPU detected via torch. Utilizing GPU (device 0).")
            return "0"
            
        logger.info("No GPU detected. Falling back to CPU.")
        return "cpu"

    def _get_pretrained_path(self) -> str:
        """Resolves pretrained model source from configuration.
        
        Returns:
            str: Ultralytics model name or validated local path.

        Raises:
            FileNotFoundError: If configured local model path is not found.
        """
        pretrained_value = str(
            self.model_config.get("pretrained_path", "yolo11n.pt")
        ).strip()

        if self._is_ultralytics_model_name(pretrained_value):
            logger.info(
                "Using Ultralytics model name directly for auto-download: "
                f"{pretrained_value}"
            )
            return pretrained_value

        local_model_path = Path(pretrained_value).expanduser().resolve()
        if not local_model_path.is_file():
            error_msg = (
                "Configured local pretrained model path does not exist: "
                f"{local_model_path}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        return str(local_model_path)

    def _is_ultralytics_model_name(self, model_value: str) -> bool:
        """Checks if configured model value is a canonical Ultralytics model name.

        Args:
            model_value (str): Model configuration value.

        Returns:
            bool: True if value should be treated as Ultralytics model name.
        """
        model_path = Path(model_value)
        has_path_hint = model_path.is_absolute() or model_path.parent != Path(".")

        if has_path_hint:
            return False

        return model_path.suffix.lower() == ".pt" and model_path.name.lower().startswith(
            "yolo"
        )

    def _get_best_model_path(self, model: YOLO, experiments_dir: str) -> Path:
        """Locates the path to the best generated model after training.
        
        Args:
            model (YOLO): The trained YOLO model instance.
            experiments_dir (str): Base directory where results are saved.
            
        Returns:
            Path: Path to the best.pt file.
            
        Raises:
            FileNotFoundError: If best.pt cannot be located.
        """
        # Try to rely on the trainer's exact save directory attribute if available
        trainer = getattr(model, "trainer", None)
        if trainer and hasattr(trainer, "save_dir"):
            best_pt = Path(trainer.save_dir) / "weights" / "best.pt"
        else:
            # Fallback path reconstruction based on project and name parameters
            # Assumes exist_ok=True or naming didn't increment (i.e. name1, name2)
            project_dir = Path(experiments_dir) / self.experiment_name
            best_pt = project_dir / "weights" / "best.pt"
            
        if not best_pt.exists():
            error_msg = f"Cannot find the best model weights at expected path: {best_pt}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Best model successfully located at: {best_pt}")
        return best_pt
