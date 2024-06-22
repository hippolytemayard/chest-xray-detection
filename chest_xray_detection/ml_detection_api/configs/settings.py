import logging

log_format = "%(asctime)s : %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

SERVER_NAME = "ChestDetectionAPI"
