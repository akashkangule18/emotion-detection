import logging
# definig logger object
logger = logging.getLogger('Sneha')
logger.setLevel(logging.DEBUG)

# getting file handler
file_handler = logging.FileHandler('akki')
file_handler.setLevel(logging.DEBUG)

# getting formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# setting formatter to handler
file_handler.setFormatter(formatter)

# adding handler to  logger
logger.addHandler(file_handler)
