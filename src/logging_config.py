
"""Central logging configuration for the application."""

import logging
from pathlib import Path


def configure_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Avoid duplicate handlers when Streamlit reruns the application.
    if not any(
        getattr(handler, "_finance_app_handler", False)
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(
            log_dir / "application.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler._finance_app_handler = True
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._finance_app_handler = True
        root_logger.addHandler(console_handler)
