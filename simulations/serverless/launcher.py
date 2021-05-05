import sys
sys.path.append("../../gym")
import numpy as np
import gym
from logger import Logger
from fixed_rm import fixed_rm
from greedy_rm import greedy_rm
from ensure_rm import ensure_rm
from lambda_rm_eval import lambda_rm_eval


def launch():

    # Set up logger wrapper
    logger_wrapper = Logger()

    # fixed_rm(
    #     logger_wrapper=logger_wrapper
    # )

    # greedy_rm(
    #     logger_wrapper=logger_wrapper
    # )

    # ensure_rm(
    #     logger_wrapper=logger_wrapper
    # )

    lambda_rm_eval(
        logger_wrapper=logger_wrapper
    )


if __name__ == "__main__":
    
    # Launch simulations
    launch()
    