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
    avg_duration_slo_trend,
    avg_harvest_cpu_percent_trend,
    avg_harvest_memory_percent_trend,
    slo_violation_percent_trend,
    acceleration_pecent_trend,
    timeout_num_trend,
    avg_interval_trend=None,
    avg_duration_slo_per_function_trend=None,
    loss_trend=None,
):
    # Log reward trend
    logger = logger_wrapper.get_logger("RewardTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(reward) for reward in reward_trend))

    # Log avg duration slo trend
    logger = logger_wrapper.get_logger("AvgDurationSLOTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(avg_duration_slo) for avg_duration_slo in avg_duration_slo_trend))

    # Log avg duration slo per function trend 
    if avg_duration_slo_per_function_trend is not None:
        logger = logger_wrapper.get_logger("AvgDurationSLOPerFunctionTrends", overwrite)
        logger.debug("")
        logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
        logger.debug("")
        for function_id in avg_duration_slo_per_function_trend.keys():
            logger.debug("{}:".format(function_id))
            logger.debug(','.join(str(avg_duration_slo) for avg_duration_slo in avg_duration_slo_per_function_trend[function_id]))

    # Log timeout number trend
    logger = logger_wrapper.get_logger("TimeoutNumTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(timeout_num) for timeout_num in timeout_num_trend))

    # Log avg harvest cpu percent trend
    logger = logger_wrapper.get_logger("AvgHarvestCPUPercentTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(avg_harvest_cpu_percent) for avg_harvest_cpu_percent in avg_harvest_cpu_percent_trend))

    # Log avg harvest memory percent trend
    logger = logger_wrapper.get_logger("AvgHarvestMemoryPercentTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(avg_harvest_memory_percent) for avg_harvest_memory_percent in avg_harvest_memory_percent_trend))

    # Log slo violation percent trend
    logger = logger_wrapper.get_logger("SLOViolationPercentTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(slo_violation_percent) for slo_violation_percent in slo_violation_percent_trend))

    # Log acceleration percent trend
    logger = logger_wrapper.get_logger("AccelerationPercentTrends", overwrite)
    logger.debug("")
    logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
    logger.debug(','.join(str(acceleration_pecent) for acceleration_pecent in acceleration_pecent_trend))

    # Log avg interval trend
    if avg_interval_trend is not None:
        logger = logger_wrapper.get_logger("AvgIntervalTrends", overwrite)
        logger.debug("")
        logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.debug("RM {}, EXP {}".format(rm_name, exp_id))
        logger.debug(','.join(str(avg_interval) for avg_interval in avg_interval_trend))

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
