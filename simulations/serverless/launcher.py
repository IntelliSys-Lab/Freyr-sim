import sys
sys.path.append("../../gym")
import numpy as np
import gym
from gym.envs.serverless.faas_params import EnvParameters, TimetableParameters
from workload_generator import WorkloadGenerator
from logger import Logger
from fixed_rm import fixed_rm
from greedy_rm import greedy_rm
from ensure_rm import ensure_rm
<<<<<<< HEAD
from lambda_rm_train import lambda_rm_train
=======
>>>>>>> 396338cac6ea37244761b01a938a2a8d4a56f49c
from lambda_rm_eval import lambda_rm_eval
import params


def launch():

    # Set up logger wrapper
    logger_wrapper = Logger()

    # Generate workload
    workload_generator = WorkloadGenerator()
    profile, timetable = workload_generator.generate_workload(
        default=params.workload_type,
<<<<<<< HEAD
        max_timestep=params.max_timestep,
=======
        max_timestep=60,
>>>>>>> 396338cac6ea37244761b01a938a2a8d4a56f49c
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="sampled_memory_traces_0.csv",
        duration_traces_file="sampled_duration_traces_0.csv",
        invocation_traces_file="sampled_invocation_traces_0.csv"
    )
    
    # Set paramters for FaaSEnv
    env_params = EnvParameters(
        max_function=params.max_function,
        max_server=params.max_server,
        cluster_size=params.cluster_size,
        user_cpu_per_server=params.user_cpu_per_server,
        user_memory_per_server=params.user_memory_per_server,
        keep_alive_window_per_server=params.keep_alive_window_per_server,
        cpu_cap_per_function=params.cpu_cap_per_function,
        memory_cap_per_function=params.memory_cap_per_function,
        timeout_penalty=params.timeout_penalty,
        interval=params.interval,
    )
    
<<<<<<< HEAD
    # Start simulations
    episode = 1

=======
    # Define episode
    episode = 3

    # FixedRM
>>>>>>> 396338cac6ea37244761b01a938a2a8d4a56f49c
    fixed_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=episode,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    # GreedyRM
    greedy_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=episode,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    # Ensure
    ensure_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=episode,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    # LambdaRM
    lambda_rm_eval(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=episode,
        hidden_dims=params.hidden_dims,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        ppo_clip=params.ppo_clip,
        ppo_epoch=params.ppo_epoch,
        value_loss_coef=params.value_loss_coef,
        entropy_coef=params.entropy_coef,
        checkpoint_path=params.model_save_path,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )


if __name__ == "__main__":
    
    # Launch simulations
    launch()
    
    
