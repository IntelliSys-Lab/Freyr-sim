import time
import functools


#
# Class utilities
#

@functools.total_ordering
class Prioritize:
    """
    Wrapper for non-comparative objects
    """
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority

#
# Function utilities
#

def log_trends(
    logger_wrapper,
    rm_name,
    overwrite,
    reward_trend,
    avg_completion_time_trend,
    avg_completion_time_per_function_trend,
    timeout_num_trend,
    loss_trend=None,
):
    # Log reward trend
    logger = logger_wrapper.get_logger("RewardTrends", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{}:".format(rm_name))
    logger.debug(','.join(str(reward) for reward in reward_trend))

    # Log avg completion time trend
    logger = logger_wrapper.get_logger("AvgCompletionTimeTrends", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{}:".format(rm_name))
    logger.debug(','.join(str(avg_completion_time) for avg_completion_time in avg_completion_time_trend))

    # Log avg completion time per function trend 
    logger = logger_wrapper.get_logger("AvgCompletionTimePerFunctionTrends", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{}:".format(rm_name))
    logger.debug("")
    for function_id in avg_completion_time_per_function_trend.keys():
        logger.debug("{}:".format(function_id))
        logger.debug(','.join(str(avg_completion_time) for avg_completion_time in avg_completion_time_per_function_trend[function_id]))

    # Log timeout number trend
    logger = logger_wrapper.get_logger("TimeoutNumTrends", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{}:".format(rm_name))
    logger.debug(','.join(str(timeout_num) for timeout_num in timeout_num_trend))

    # Log loss trend
    if loss_trend is not None:
        logger = logger_wrapper.get_logger("LossTrends", overwrite)
        logger.debug("")
        logger.debug("**********")
        logger.debug("**********")
        logger.debug("**********")
        logger.debug("")
        logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.debug("{}:".format(rm_name))
        logger.debug(','.join(str(loss) for loss in loss_trend))

    # Rollback handler 
    logger = logger_wrapper.get_logger(rm_name, False)
    

def log_resource_utils(
    logger_wrapper,
    overwrite,
    rm_name,
    episode,
    resource_utils_record
):
    logger = logger_wrapper.get_logger("ResourceUtils", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{} episode {}:".format(rm_name, episode))

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
    logger_wrapper,
    overwrite,
    rm_name,
    episode,
    function_throughput_list
):
    logger = logger_wrapper.get_logger("FunctionThroughput", overwrite)
    logger.debug("")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("**********")
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("{} episode {}:".format(rm_name, episode))
    logger.debug("function_throughput")
    logger.debug(','.join(str(function_throughput) for function_throughput in function_throughput_list))

    # Rollback handler 
    logger = logger_wrapper.get_logger(rm_name, False)