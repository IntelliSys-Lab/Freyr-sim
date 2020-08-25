import numpy as np
import os
import logging


timestep = 100

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

log_path = os.path.dirname(os.getcwd()) + '/serverless/logs/'
log_name = log_path + "test_timestep_{}.txt".format(timestep)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_name, mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

for i in range(timestep):
    logger.debug("No bug")
    logger.info("Timestep: {}".format(i))
    