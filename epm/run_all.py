"""Orchestrator for EPM sub-project."""

import argparse
import logging
from epm.upload_supabase import main as upload_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("Running EPM pipeline...")
    upload_main()

if __name__ == "__main__":
    main()
