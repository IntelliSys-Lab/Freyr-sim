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
from lambda_rm_train import lambda_rm_train
from lambda_rm_eval import lambda_rm_eval


def launch():

    # Set up logger wrapper
    logger_wrapper = Logger()

    # Generate workload
    workload_generator = WorkloadGenerator()
    profile, timetable = workload_generator.generate_workload(
        default="azure",
        max_timestep=60,
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="sampled_memory_traces_0.csv",
        duration_traces_file="sampled_duration_traces_0.csv",
        invocation_traces_file="sampled_invocation_traces_0.csv"
    )
    
    # Set paramters for FaaSEnv
    env_params = EnvParameters(
        max_function=110,
        max_server=20,
        cluster_size=20,
        user_cpu_per_server=32,
        user_memory_per_server=32,
        keep_alive_window_per_server=10,
        cpu_cap_per_function=16,
        memory_cap_per_function=16,
        interval=1,
        timeout_penalty=600
    )
    
    # Start simulations
    fixed_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=1,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    greedy_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=1,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    ensure_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=1,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    lambda_rm_eval(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=10,
        hidden_dims=[32, 16],
        learning_rate=0.001,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_epoch=5,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        checkpoint_path="ckpt/best_model.pth",
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )


if __name__ == "__main__":
    
    # Launch simulations
    launch()
    
    
