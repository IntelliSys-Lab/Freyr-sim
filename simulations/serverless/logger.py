import os
import logging


def get_logger(file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_path = os.path.dirname(os.getcwd()) + '/serverless/logs/'
    log_name = log_path + "{}.txt".format(file_name)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
    