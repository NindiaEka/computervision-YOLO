from pipelines.inference_pipeline import InferencePipeline
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


def main() -> None:
    """Runs the inference pipeline CLI entry point.

    This function only initializes dependencies, executes the pipeline,
    and logs top-level process status.
    """
    logger.info("=" * 40)
    logger.info("Starting Inference Application...")
    logger.info("=" * 40)

    try:
        config_loader = ConfigLoader()
        pipeline = InferencePipeline(config_loader=config_loader)
        summary = pipeline.run()

        logger.info("=" * 40)
        if summary.status == "success":
            logger.info("Inference Completed Successfully")
        elif summary.status == "partial_failed":
            logger.info("Inference Completed With Warnings")
        else:
            logger.info("Inference Failed")
        logger.info("=" * 40)
        logger.info(f"Run ID          : {summary.run_id}")
        logger.info(f"Model           : {summary.model_name}")
        logger.info(f"Videos Processed: {summary.total_videos}")
        logger.info(f"Successful Videos: {summary.successful_videos}")
        logger.info(f"Failed Videos   : {summary.failed_videos}")
        logger.info(f"Total Frames    : {summary.total_frames}")
        logger.info(f"Processed Frames: {summary.processed_frames}")
        logger.info(f"Total Detections: {summary.total_detections}")
        logger.info(f"Speed           : {summary.inference_fps} FPS")
        logger.info(f"Processing Time : {summary.processing_time}")
        logger.info(f"Output Folder   : {summary.output_folder}")
        logger.info("=" * 40)
    except Exception as exc:
        logger.exception(f"Inference application failed: {exc}")
        raise


if __name__ == "__main__":
    main()
