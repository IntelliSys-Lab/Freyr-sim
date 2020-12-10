import sys
sys.path.append("../../gym")
import numpy as np
import gym
from gym.envs.serverless.faas_params import EnvParameters, TimetableParameters
from workload_generator import WorkloadGenerator
from logger import Logger
from fixed_rm import fixed_rm
from greedy_rm import greedy_rm
from lambda_rm import lambda_rm_train, lambda_rm_eval



def launch():

    # Set up logger wrapper
    logger_wrapper = Logger()

    # Generate workload
    workload_generator = WorkloadGenerator()
    
    timetable_params = TimetableParameters(
        max_timestep=60,
        distribution_type="mod",
        mod_factors=[1, 1, 1, 1, 2, 2, 4, 4, 6, 8]
    )
    # timetable_params = TimetableParameters(
    #     max_timestep=60,
    #     distribution_type="bernoulli",
    #     bernoulli_p=0.5
    # )
    # timetable_params = TimetableParameters(
    #     max_timestep=60,
    #     distribution_type="poisson",
    #     poisson_mu=0.8
    # )
    # timetable_params = None # Azure traces
    
    profile, timetable = workload_generator.generate_workload(
        # default="azure",
        default="ensure",
        timetable_params=timetable_params
    )
    
    # Set paramters for FaaSEnv
    env_params = EnvParameters(
        cluster_size=10,
        user_cpu_per_server=8,
        user_memory_per_server=8,
        keep_alive_window_per_server=60,
        cpu_cap_per_function=8,
        memory_cap_per_function=8,
        timeout_penalty=60,
        interval=1,
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

    # greedy_provision(
    #     profile=profile,
    #     timetable=timetable,
    #     env_params=env_params,
    #     max_episode=10,
    #     save_plot=True,
    #     show_plot=False,
    #     logger_wrapper=logger_wrapper
    # )

    lambda_rm_train(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=1000,
        hidden_dims=[32, 16],
        learning_rate=0.001,
        discount_factor=1,
        ppo_clip=0.2,
        ppo_steps=5,
        model_save_path="ckpt/best_model.pth",
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
        ppo_steps=5,
        checkpoint_path="ckpt/best_model.pth",
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )


if __name__ == "__main__":
    
    # Launch simulations
    launch()
    
    
