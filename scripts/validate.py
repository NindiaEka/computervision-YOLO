from pipelines.validation_pipeline import ValidationPipeline
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


def main() -> None:
    """Runs the validation pipeline CLI entry point.

    This function only initializes dependencies, executes the pipeline,
    and logs top-level process status.
    """
    logger.info("=" * 40)
    logger.info("Starting Validation Application...")
    logger.info("=" * 40)

    try:
        config_loader = ConfigLoader()
        pipeline = ValidationPipeline(config_loader=config_loader)
        summary = pipeline.run()

        logger.info("=" * 40)
        logger.info("Validation Completed Successfully")
        logger.info("=" * 40)
        logger.info(f"Run ID          : {summary.run_id}")
        logger.info(f"Model           : {summary.model_name}")
        logger.info(f"Dataset         : {summary.dataset}")
        logger.info("")
        logger.info(f"Precision       : {summary.metrics.precision}")
        logger.info(f"Recall          : {summary.metrics.recall}")
        logger.info(f"mAP50           : {summary.metrics.map50}")
        logger.info(f"mAP50-95        : {summary.metrics.map50_95}")
        logger.info("")
        logger.info(f"Processing Time : {summary.processing_time}")
        logger.info(f"Output Folder   : {summary.output_folder}")
        logger.info("=" * 40)
    except Exception as exc:
        logger.exception(f"Validation application failed: {exc}")
        raise


if __name__ == "__main__":
    main()
