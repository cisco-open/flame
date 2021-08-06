import logging
import os
import sys

log_path = '/tmp/fledge-job.log'

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format=
    '%(asctime)s | %(name)s:%(lineno)d | %(levelname)s | %(threadName)s | %(message)s',
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler(sys.stdout)]
)
