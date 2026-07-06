import logging
from pathlib import Path
from typing import Dict, Any, List

import yaml

from utils.logger import Logger

logger = Logger.get_logger(__name__)


class DatasetService:
    """Service to validate and summarize YOLO format datasets.

    This service ensures that a dataset downloaded or provided manually
    is structurally correct, has valid YAML configuration, and is ready
    for model training.

    Attributes:
        yaml_path (Path): Path to the dataset's data.yaml configuration file.
        dataset_dir (Path): The root directory of the dataset.
        data (Dict[str, Any]): The loaded YAML contents.
        is_valid (bool): Flag indicating if the dataset has passed validation.
    """

    def __init__(self, yaml_path: str) -> None:
        """Initializes the DatasetService with a target data.yaml path.

        Args:
            yaml_path (str): The absolute or relative path to the data.yaml file.
        """
        self.yaml_path = Path(yaml_path)
        self.dataset_dir = self.yaml_path.parent
        self.data: Dict[str, Any] = {}
        self.is_valid = False

    def validate(self) -> bool:
        """Validates the dataset structure and configuration.

        Ensures the data.yaml exists, has valid syntax, contains required 
        YOLO subsets (train, val), configures class names correctly, and 
        verifies that subset directories are present.

        Returns:
            bool: True if the dataset is valid and ready for training, 
                  False otherwise.
        """
        try:
            self._load_yaml()
            self._validate_keys()
            self._validate_directories()
            
            self.is_valid = True
            logger.info(f"Dataset at '{self.dataset_dir}' is valid and ready for training.")
        except Exception as e:
            self.is_valid = False
            logger.error(f"Dataset validation failed: {str(e)}")

        return self.is_valid

    def summary(self) -> Dict[str, Any]:
        """Generates and logs a complete summary of the dataset.

        Returns:
            Dict[str, Any]: A dictionary containing metadata such as subset image counts,
            total classes, and class names.

        Raises:
            RuntimeError: If the dataset has not been validated successfully.
        """
        if not self.is_valid:
            logger.warning("Dataset hasn't been validated yet. Running validation...")
            if not self.validate():
                raise RuntimeError("Cannot summarize an invalid dataset.")

        summary_data: Dict[str, Any] = {
            "classes_count": self.data.get("nc", 0),
            "class_names": self._get_class_names(),
            "subsets": {}
        }

        # Check standard YOLO subsets
        for subset in ["train", "val", "test"]:
            if subset in self.data:
                subset_path = self._resolve_path(self.data[subset])
                count = self._count_images(subset_path)
                summary_data["subsets"][subset] = count

        self._log_summary(summary_data)
        return summary_data

    def _load_yaml(self) -> None:
        """Loads and parses the data.yaml file.

        Raises:
            FileNotFoundError: If the data.yaml file does not exist.
            ValueError: If the YAML content is unreadable or empty.
        """
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found at '{self.yaml_path}'")

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as file:
                self.data = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML formatting: {exc}")

        if not self.data:
            raise ValueError("The data.yaml file is empty.")

    def _validate_keys(self) -> None:
        """Validates the presence of required YOLO dataset keys.

        Raises:
            ValueError: If mandatory keys like 'train', 'val', 'nc', or 'names' are missing.
        """
        required_keys = ["train", "val", "nc", "names"]
        missing_keys = [key for key in required_keys if key not in self.data]

        if missing_keys:
            raise ValueError(f"Missing required keys in data.yaml: {missing_keys}")

        if not isinstance(self.data.get("nc"), int) or self.data["nc"] <= 0:
            raise ValueError("'nc' (number of classes) must be a positive integer.")

    def _validate_directories(self) -> None:
        """Validates the existence of configured subset directories.
        
        Note: YOLO strictly uses 'val' internally, though sometimes 'valid' 
        is mapped, we enforce 'train' and 'val' as mandatory per generic YOLO formats.

        Raises:
            ValueError: If subset paths are not defined properly.
            FileNotFoundError: If the subset folders are missing from the filesystem.
        """
        # Train & Val are mandatory for YOLO training
        for subset in ["train", "val"]:
            path_str = self.data.get(subset)
            if not path_str:
                raise ValueError(f"'{subset}' path is missing in data.yaml.")

            resolved_path = self._resolve_path(path_str)
            if not resolved_path.is_dir():
                raise FileNotFoundError(f"Subset directory '{subset}' not found at '{resolved_path}'")

        # Test is optional
        if "test" in self.data:
            test_path = self._resolve_path(self.data["test"])
            if not test_path.is_dir():
                logger.warning(f"Test directory is defined but not found at '{test_path}'")

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolves a subset path relative to the dataset root.

        Args:
            relative_path (str): The path read from data.yaml.

        Returns:
            Path: The fully resolved absolute path.
        """
        path = Path(relative_path)
        if path.is_absolute():
            return path
            
        # Typically YOLO datasets structure paths relative to the yaml file's location
        return (self.dataset_dir / path).resolve()

    def _count_images(self, directory: Path) -> int:
        """Recursively counts the number of accepted image files inside a directory.

        Args:
            directory (Path): The target directory.

        Returns:
            int: The total count of image files.
        """
        if not directory.exists() or not directory.is_dir():
            return 0
            
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        
        # rglob handles nested structures like train/images/ just in case
        return sum(
            1 for file in directory.rglob("*.*") 
            if file.suffix.lower() in image_extensions
        )

    def _get_class_names(self) -> List[str]:
        """Safely parses and retrieves class names from the dataset configuration.

        Returns:
            List[str]: A list of class name strings.
        """
        names = self.data.get("names", [])
        
        # In newer YOLO formats, names is often a dictionary {0: "car", 1: "bike"}
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys())]
        
        # In older formats, names is a list ["car", "bike"]
        if isinstance(names, list):
            return [str(name) for name in names]
            
        return []

    def _log_summary(self, summary_data: Dict[str, Any]) -> None:
        """Logs the formatted dataset summary.

        Args:
            summary_data (Dict[str, Any]): The structured summary metrics.
        """
        logger.info("\n" + "="*50)
        logger.info(" DATASET SUMMARY ".center(50))
        logger.info("="*50)
        logger.info(f"Dataset Root   : {self.dataset_dir}")
        logger.info(f"Total Classes  : {summary_data['classes_count']}")
        
        class_names_str = ", ".join(summary_data['class_names'])
        logger.info(f"Class Names    : {class_names_str}")
        
        logger.info("-" * 50)
        logger.info("Image Counts by Subset:")
        
        for subset, count in summary_data['subsets'].items():
            logger.info(f"  - {subset.upper():<5} : {count} images")
            
        logger.info("="*50 + "\n")
