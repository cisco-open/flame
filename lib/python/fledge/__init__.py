import logging
import sys

log_path = '/tmp/fledge-job.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler(sys.stdout)]
)
