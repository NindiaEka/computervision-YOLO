import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """A utility class providing a customized, reusable logger for the framework.

    This class configures the built-in Python logging module to output
    messages to both the console and a file using a standard format.
    """

    @staticmethod
    def get_logger(
        name: str, 
        log_file: Optional[str] = "experiments/framework.log", 
        level: int = logging.INFO
    ) -> logging.Logger:
        """Retrieves and configures a logger instance.

        Configures the logger with a custom format if handlers are not
        already attached, preventing duplicate logs. By default, it logs
        both to the console and to a specified log file.

        Args:
            name (str): The name of the logger, typically __name__ from the calling module.
            log_file (Optional[str]): The path to the file where logs will be saved.
                Defaults to "experiments/framework.log". Set to None to disable file logging.
            level (int): The logging level threshold (e.g., logging.INFO, logging.DEBUG).
                Defaults to logging.INFO.

        Returns:
            logging.Logger: A configured logger instance ready for use.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # To prevent adding duplicate handlers if the logger is requested multiple times
        if not logger.handlers:
            # Format: [2026-07-06 09:15:23] INFO Training started
            formatter = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Console Handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File Handler
            if log_file:
                log_path = Path(log_file)
                # Ensure the parent directory exists
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        return logger
