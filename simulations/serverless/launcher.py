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
        max_timestep=600,
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="simple_memory_traces.csv",
        duration_traces_file="simple_duration_traces.csv",
        invocation_traces_file="simple_invocation_traces.csv"
    )
    
    # Set paramters for FaaSEnv
    env_params = EnvParameters(
        max_function=500,
        max_server=100,
        cluster_size=30,
        user_cpu_per_server=64,
        user_memory_per_server=64,
        keep_alive_window_per_server=60,
        cpu_cap_per_function=8,
        memory_cap_per_function=8,
        interval=1,
        timeout_penalty=600
    )
    
    # Start simulations
    fixed_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=10,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    greedy_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=10,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    ensure_rm(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=10,
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
    
    
