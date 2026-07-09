from pipelines.video_processing_pipeline import VideoProcessingPipeline
from utils.config import ConfigLoader
from utils.logger import Logger


logger = Logger.get_logger(__name__)


def main() -> None:
    """Runs the video processing pipeline CLI entry point.

    This function only initializes dependencies, executes the pipeline,
    and logs top-level process status.
    """
    logger.info("=" * 40)
    logger.info("Starting Video Processing Application...")
    logger.info("=" * 40)

    try:
        config_loader = ConfigLoader()
        pipeline = VideoProcessingPipeline(config_loader=config_loader)
        summary = pipeline.run()

        logger.info("=" * 40)
        logger.info("Video Processing Completed Successfully")
        logger.info("=" * 40)
        logger.info(f"Run ID          : {summary.run_id}")
        logger.info(f"Total Videos    : {summary.total_videos}")
        logger.info(f"Total Frames    : {summary.total_frames}")
        logger.info(f"Processing Time : {summary.processing_time}")
    except Exception as exc:
        logger.exception(f"Video processing application failed: {exc}")
        raise


if __name__ == "__main__":
    main()
