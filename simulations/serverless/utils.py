import time
import numpy as np


#
# Function utilities
#

def log_trends(
    overwrite,
    rm_name,
    exp_id,
    logger_wrapper,
    reward_trend,
    avg_completion_time_slo_trend,
    avg_completion_time_trend,
    timeout_num_trend,
    avg_trends_per_function=None,
    loss_trend=None,
):
    # Log reward trend
    logger = logger_wrapper.get_logger("RewardTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(reward) for reward in reward_trend))

    # Log avg completion time SLO trend
    logger = logger_wrapper.get_logger("AvgCompletionTimeSLOTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(avg_completion_time_slo) for avg_completion_time_slo in avg_completion_time_slo_trend))

    # Log avg completion time trend
    logger = logger_wrapper.get_logger("AvgCompletionTimeTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(avg_completion_time) for avg_completion_time in avg_completion_time_trend))

    # Log avg completion time per function trend 
    if avg_trends_per_function is not None:
        logger = logger_wrapper.get_logger("AvgPerFunctionTrends", overwrite)
        logger.debug("")
        logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
        logger.debug("")
        logger.debug("Function, Avg Completion Time, Avg Completion Time SLO, Avg CPU SLO, Avg Memory SLO")
        for function_id in avg_trends_per_function.keys():
            avg_completion_time = np.mean(avg_trends_per_function[function_id]["avg_completion_time"])
            avg_completion_time_slo = np.mean(avg_trends_per_function[function_id]["avg_completion_time_slo"])
            avg_cpu_slo = np.mean(avg_trends_per_function[function_id]["avg_cpu_slo"])
            avg_memory_slo = np.mean(avg_trends_per_function[function_id]["avg_memory_slo"])
            logger.debug("{},{},{},{},{}".format(function_id, avg_completion_time, avg_completion_time_slo, avg_cpu_slo, avg_memory_slo))

    # Log timeout number trend
    logger = logger_wrapper.get_logger("TimeoutNumTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(timeout_num) for timeout_num in timeout_num_trend))

    # Log loss trend
    if loss_trend is not None:
        logger = logger_wrapper.get_logger("LossTrends", overwrite)
        logger.debug("")
        logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
        logger.debug(','.join(str(loss) for loss in loss_trend))

    # Rollback handler 
    logger = logger_wrapper.get_logger(rm_name, False)


def log_resource_utils(
    overwrite,
    rm_name,
    exp_id,
    logger_wrapper,
    episode,
    resource_utils_record
):
    logger = logger_wrapper.get_logger("ResourceUtils", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}, Episode {}:".format(rm_name, exp_id, episode))

    n_server = len(resource_utils_record) - 1

    for i in range(n_server):
        server = "server{}".format(i)

        logger.debug(server)
        logger.debug("cpu_util")
        logger.debug(','.join(str(cpu_util) for cpu_util in resource_utils_record[server]["cpu_util"]))
        logger.debug("memory_util")
        logger.debug(','.join(str(memory_util) for memory_util in resource_utils_record[server]["memory_util"]))
        logger.debug("avg_cpu_util")
        logger.debug(resource_utils_record[server]["avg_cpu_util"])
        logger.debug("avg_memory_util")
        logger.debug(resource_utils_record[server]["avg_memory_util"])
        logger.debug("")

    logger.debug("avg_server")
    logger.debug("cpu_util")
    logger.debug(','.join(str(cpu_util) for cpu_util in resource_utils_record["avg_server"]["cpu_util"]))
    logger.debug("memory_util")
    logger.debug(','.join(str(memory_util) for memory_util in resource_utils_record["avg_server"]["memory_util"]))
    logger.debug("avg_cpu_util")
    logger.debug(resource_utils_record["avg_server"]["avg_cpu_util"])
    logger.debug("avg_memory_util")
    logger.debug(resource_utils_record["avg_server"]["avg_memory_util"])

    # Rollback handler 
    logger = logger_wrapper.get_logger(rm_name, False)


def log_function_throughput(
    overwrite,
    rm_name,
    exp_id,
    logger_wrapper,
    episode,
    function_throughput_list
):
    logger = logger_wrapper.get_logger("FunctionThroughput", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}, Episode {}:".format(rm_name, exp_id, episode))
    logger.debug("function_throughput")
    logger.debug(','.join(str(function_throughput) for function_throughput in function_throughput_list))

    # Rollback handler 
    logger = logger_wrapper.get_logger(rm_name, False)
