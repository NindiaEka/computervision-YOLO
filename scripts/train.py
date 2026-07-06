from services.training_pipeline import TrainingPipeline
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


def main() -> None:
    """Runs the training pipeline entry point.

    This function only initializes dependencies and executes the pipeline.
    """
    logger.info("Starting training application...")

    try:
        config_loader = ConfigLoader()
        pipeline = TrainingPipeline(config_loader=config_loader)
        pipeline.run()
        logger.info("Training application finished successfully.")
    except Exception as exc:
        logger.exception(f"Training application failed: {exc}")
        raise


if __name__ == "__main__":
    main()
