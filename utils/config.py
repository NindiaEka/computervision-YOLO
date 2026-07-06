from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.logger import Logger

logger = Logger.get_logger(__name__)


class ConfigLoader:
    """A class to load and parse YAML configuration files.

    Attributes:
        config_path (str): The path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """Initializes the ConfigLoader with the specified config path.

        Args:
            config_path (str): The path to the configuration file.
                Defaults to "configs/config.yaml".
        """
        self.config_path = config_path
        self._config_cache: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed configuration data.

        Raises:
            FileNotFoundError: If the specified configuration file is not found.
            yaml.YAMLError: If the configuration file contains invalid YAML formatting.
        """
        if self._config_cache is not None:
            return self._config_cache

        path = Path(self.config_path)

        if not path.is_file():
            error_msg = f"Configuration file not found at: {self.config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                logger.info(f"Successfully loaded configuration from '{self.config_path}'")
                
                # Handle empty YAML files gracefully
                if config_data is None:
                    logger.warning(f"Configuration file '{self.config_path}' is empty.")
                    self._config_cache = {}
                    return self._config_cache
                    
                self._config_cache = config_data
                return self._config_cache
        except yaml.YAMLError as exc:
            error_msg = f"Invalid YAML formatting in '{self.config_path}': {exc}"
            logger.error(error_msg)
            raise

    def get_config(self) -> Dict[str, Any]:
        """Returns the cached configuration dictionary.

        Returns:
            Dict[str, Any]: Cached configuration dictionary.

        Raises:
            RuntimeError: If configuration has not been loaded yet.
        """
        if self._config_cache is None:
            error_msg = (
                "Configuration has not been loaded yet. "
                "Load configuration in the composition root first."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return self._config_cache
