import sys
sys.path.append("../../gym")
import numpy as np
import gym
from gym.envs.serverless.faas_params import EnvParameters, TimetableParameters
from workflow_generator import WorkflowGenerator

from fixed_provision import fixed_provision
from greedy_provision import greedy_provision
from pg_provision import pg_provision



def launch():

    # Generate workflow
    workflow_generator = WorkflowGenerator()
    
    timetable_params = TimetableParameters(max_timestep=200,
                                           distribution_type="mod",
                                           mod_factors=[1, 1, 30, 30]
                                           )
#     timetable_params = TimetableParameters(max_timestep=200,
#                                            distribution_type="bernoulli",
#                                            bernoulli_p=0.5
#                                            )
#     timetable_params = TimetableParameters(max_timestep=200,
#                                            distribution_type="poisson",
#                                            poisson_mu=0.8
#                                            )
    
    profile, timetable = workflow_generator.generate_workflow(timetable_params=timetable_params)
    
    # Set paramters for FaaSEnv
    env_params = EnvParameters(
        cpu_total=32*1000,
        memory_total=45*1000,
        cpu_cap_per_function=32,
        memory_cap_per_function=45
        )
    
    # Number of max episode
    max_episode = 500
    
    # Start simulations
#     fixed_provision(profile=profile,
#                     timetable=timetable,
#                     env_params=env_params,
#                     max_episode=max_episode,
#                     plot_prefix_name="Fixed_Mod_No_Timeout",
#                     save_plot=True,
#                     show_plot=False
#                     )
#     greedy_provision(profile=profile,
#                      timetable=timetable,
#                      env_params=env_params,
#                      max_episode=max_episode,
#                      plot_prefix_name="Greedy_Mod_No_Timeout",
#                      save_plot=True,
#                      show_plot=False
#                      )
    pg_provision(profile=profile,
                 timetable=timetable,
                 env_params=env_params,
                 max_episode=max_episode,
                 plot_prefix_name="PG_Mod_No_Timeout",
                 save_plot=True,
                 show_plot=False
                 )
    


if __name__ == "__main__":
    
    # Launch simulations
    launch()
    
    