# From https://gist.github.com/nguyendv/ccf4d9e1d0b4679da938872378c46226

import logging
from logging.handlers import RotatingFileHandler

def setup_logger(logdir="log", prefix=""):
  MAX_BYTES = 10000000 # Maximum size for a log file
  BACKUP_COUNT = 9 # Maximum number of old log files
  
  # The name should be unique, so you can get in in other places
  # by calling `logger = logging.getLogger('com.dvnguyen.logger.example')
  logger = logging.getLogger('com.dvnguyen.logger.example') 
  logger.setLevel(logging.INFO) # the level should be the lowest level set in handlers

  log_format = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s")

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(log_format)
  stream_handler.setLevel(logging.INFO)
  logger.addHandler(stream_handler)

  info_handler = RotatingFileHandler(f'{logdir}/{prefix}info.log', maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
  info_handler.setFormatter(log_format)
  info_handler.setLevel(logging.INFO)
  logger.addHandler(info_handler)

  error_handler = RotatingFileHandler(f'{logdir}/{prefix}error.log', maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
  error_handler.setFormatter(log_format)
  error_handler.setLevel(logging.ERROR)
  logger.addHandler(error_handler)
  return logger
  
if __name__ == '__main__':
  logger = setup_logger()
  for i in range(0, 1000):
    logger.info('This is a message {}'.format(i))
    if i % 5 == 0:
      logger.error('THis is a error {}'.format(i))