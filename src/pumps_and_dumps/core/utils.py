import logging
import os
import warnings


def available_cpu_count() -> int:
    """Count CPUs available to this process, with a one-worker fallback."""
    return os.process_cpu_count() or os.cpu_count() or 1


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)
