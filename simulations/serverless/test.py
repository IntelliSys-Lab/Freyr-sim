import numpy as np
import torch


def encode_action(function_profile, resource_adjust_list):
    actions = []
    
    for function in function_profile:
        for key in resource_adjust_list.keys():
            if function.function_id == key:
                index = function_profile.index(function)
                
                if resource_adjust_list[key][0] != 0:
                    adjust_cpu = index*4 + resource_adjust_list[key][0]
                    actions.append(adjust_cpu)
                if resource_adjust_list[key][1] != 0:
                    adjust_memory = index*4 + resource_adjust_list[key][1]
                    actions.append(adjust_memory)
                    
    return actions

class Func():
    
    def __init__(self, function_id):
        self.function_id = function_id

function_profile = [Func("1"), Func("2"), Func("3"), Func("4")]
resource_adjust_list = {"1": [1, 3],
                        "2": [0, 0],
                        "3": [0, 0],
                        "4": [1, 3]}

actions = encode_action(function_profile, resource_adjust_list)

print(actions)

