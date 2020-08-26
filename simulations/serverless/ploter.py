import matplotlib.pyplot as plt
import os


class Ploter():
    """
    Plot trend for each episode
    """
    def __init__(self,
                 fig_path=os.path.dirname(os.getcwd())+"/serverless/figures/"
                 ):
        self.fig_path = fig_path
    
    def plot_save(self,
                  policy_name,
                  reward_trend, 
                  avg_slow_down_trend, 
                  timeout_num_trend
                  ):
        fig_1 = plt.figure('Total Reward Trend', figsize = (6,4)).add_subplot(111)
        fig_1.plot(reward_trend)
        fig_1.set_xlabel("Episode")
        fig_1.set_ylabel("Total reward")
        plt.savefig(self.fig_path + policy_name + "_Total_Reward_Trend.png")
        plt.clf()
        
        fig_2 = plt.figure('Avg Slow Down Trend', figsize = (6,4)).add_subplot(111)
        fig_2.plot(avg_slow_down_trend)
        fig_2.set_xlabel("Episode")
        fig_2.set_ylabel("Avg slow down")
        plt.savefig(self.fig_path + policy_name + "_Avg_Slow_Down_Trend.png")
        plt.clf()
        
        fig_3 = plt.figure('Timeout Num Trend', figsize = (6,4)).add_subplot(111)
        fig_3.plot(timeout_num_trend)
        fig_3.set_xlabel("Episode")
        fig_3.set_ylabel("Timeout num")
        plt.savefig(self.fig_path + policy_name + "_Timeout_Num_Trend.png")
        plt.clf()
        
    def plot_show(self,
                  policy_name,
                  reward_trend, 
                  avg_slow_down_trend, 
                  timeout_num_trend
                  ):
        fig_1 = plt.figure('Total Reward Trend', figsize = (6,4)).add_subplot(111)
        fig_1.plot(reward_trend)
        fig_1.set_xlabel("Episode")
        fig_1.set_ylabel("Total reward")
        
        fig_2 = plt.figure('Avg Slow Down Trend', figsize = (6,4)).add_subplot(111)
        fig_2.plot(avg_slow_down_trend)
        fig_2.set_xlabel("Episode")
        fig_2.set_ylabel("Avg slow down")
        
        fig_3 = plt.figure('Timeout Num Trend', figsize = (6,4)).add_subplot(111)
        fig_3.plot(timeout_num_trend)
        fig_3.set_xlabel("Episode")
        fig_3.set_ylabel("Timeout num")
        
        plt.show()
        

