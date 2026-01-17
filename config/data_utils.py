import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s:%(lineno)d - %(message)s",
)
logging.getLogger("matplotlib.font_manager").disabled = True  # This fucker floods logs
logger = logging.getLogger(__name__)


def inspect_dataloader(dataloader, name="Train"):
    """Inspect and log dataloader batch shapes"""
    for x, y in dataloader:
        logger.debug(f"{name} batch x shape: {x.shape}")
        logger.debug(f"{name} batch y shape: {y.shape}")
        break
    logger.debug(f"Length of {name} dataloader: {len(dataloader)}")
