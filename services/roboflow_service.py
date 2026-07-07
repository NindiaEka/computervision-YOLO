import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from roboflow import Roboflow

from utils.config import ConfigLoader
from utils.logger import Logger

logger = Logger.get_logger(__name__)


class RoboflowService:
    """Service to handle automated dataset downloads from Roboflow.
    
    Attributes:
        config (Dict[str, Any]): The configuration dictionary retrieved via ConfigLoader.
        api_key (str): The Roboflow API key.
        workspace (str): The Roboflow workspace name.
        project_name (str): The Roboflow project name.
        version (int): The dataset version number.
        datasets_dir (Path): The local root directory to store downloaded datasets.
        dataset_path (Path): The exact local directory path for this specific dataset version.
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        """Initializes the RoboflowService and loads necessary configurations.
        
        Args:
            config_loader (Optional[ConfigLoader]): An instance of ConfigLoader to fetch 
                project settings. If None, instantiates a default ConfigLoader.
        """
        if config_loader is None:
            config_loader = ConfigLoader()
            
        self.config = config_loader.get_config()
        
        # Load specific configurations
        rf_config = self.config.get("roboflow", {})
        self.api_key = rf_config.get("api_key", "")
        self.workspace = rf_config.get("workspace", "")
        self.project_name = rf_config.get("project", "")
        self.version = rf_config.get("version", 1)
        
        # Define directories based on configs
        out_config = self.config.get("output", {})
        base_datasets_dir = out_config.get("datasets_dir", "datasets")
        
        self.datasets_dir = Path(base_datasets_dir)
        self.dataset_path = self.datasets_dir / f"{self.project_name}-{self.version}"

    def get_dataset(self) -> str:
        """Main method to ensure the dataset is available locally.
        
        Checks if the dataset exists locally; if not, triggers the download.
        
        Returns:
            str: Path to the data.yaml file of the dataset.
            
        Raises:
            ValueError: If Roboflow credentials are empty or dataset download fails.
            FileNotFoundError: If data.yaml is not found after download/checking.
        """
        if self._is_dataset_present():
            logger.info(f"Dataset already exists locally at: {self.dataset_path}")
        else:
            logger.info(f"Dataset not found locally. Proceeding to download...")
            self._download_dataset()
            
        yaml_path = self._get_yaml_path()
        return str(yaml_path)

    def _is_dataset_present(self) -> bool:
        """Checks whether the requested dataset already exists locally.
        
        Returns:
            bool: True if the dataset directory exists, False otherwise.
        """
        return self.dataset_path.exists() and self.dataset_path.is_dir()

    def _download_dataset(self) -> None:
        """Connects to Roboflow and downloads the dataset in YOLO11 format."""
        if not self.api_key or self.api_key == "YOUR_ROBOFLOW_API_KEY":
            error_msg = "Invalid Roboflow API key. Please check your config.yaml."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Connecting to Roboflow Workspace: '{self.workspace}'...")
        
        try:
            rf = Roboflow(api_key=self.api_key)
            project = rf.workspace(self.workspace).project(self.project_name)
            version = project.version(self.version)
            
            logger.info(f"Downloading dataset '{self.project_name}' version {self.version}...")

            self.datasets_dir.mkdir(parents=True, exist_ok=True)

            # YOLO11 format maps to 'yolov8' in Roboflow export types.
            dataset = version.download("yolov8")
            downloaded_dir = Path(dataset.location).resolve()
            target_dir = self.dataset_path.resolve()

            if downloaded_dir != target_dir:
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded_dir), str(target_dir))
                logger.info(f"Dataset moved to target directory: {target_dir}")

            self.dataset_path = target_dir
            self._normalize_dataset_yaml()
                
            logger.info("Dataset downloaded successfully.")
            
        except Exception as e:
            error_msg = f"Failed to download dataset from Roboflow: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _normalize_dataset_yaml(self) -> None:
        """Normalizes Roboflow-exported subset paths inside dataset YAML.

        Converts paths like "../train/images" into "train/images" so that
        dataset consumers can resolve paths consistently from dataset root.
        """
        yaml_path = self._get_yaml_path()

        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            logger.warning(f"Skipping YAML normalization because format is not a mapping: {yaml_path}")
            return

        normalized = False
        for key in ["train", "val", "test"]:
            value = data.get(key)
            if isinstance(value, str) and value.startswith("../"):
                data[key] = value.replace("../", "", 1)
                normalized = True

        if normalized:
            with open(yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
            logger.info(f"Normalized dataset YAML paths at: {yaml_path}")

    def _get_yaml_path(self) -> Path:
        """Retrieves and validates the path to the dataset's data.yaml configuration.
        
        Returns:
            Path: The validated path to data.yaml file.
            
        Raises:
            FileNotFoundError: If the data.yaml file cannot be located inside the dataset folder.
        """
        yaml_path = self.dataset_path / "data.yaml"
        
        if not yaml_path.is_file():
            # In some exports, the file is named project-name.yaml or similar
            yaml_files = list(self.dataset_path.glob("*.yaml"))
            if yaml_files:
                yaml_path = yaml_files[0]
                logger.warning(f"'data.yaml' not found. Using alternative YAML: {yaml_path.name}")
            else:
                error_msg = f"Cannot find any YAML configuration file in {self.dataset_path}."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
        return yaml_path
