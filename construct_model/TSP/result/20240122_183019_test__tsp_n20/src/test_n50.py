##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from construct_utils.utils import create_logger, copy_all_src
from construct_utils.cal_ps_hv import cal_ps_hv

from datetime import datetime

##########################################################################################
import time
import hvwfg
from construct_model.TSP.TSPTester import TSPTester as Tester

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.style.use('default')
##########################################################################################
# parameters

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/construct_model_TSP/50',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': False,
    'aug_factor': 1,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n50',
        'filename': 'run_log'
    }
}
##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################
# 以构造方式预构造解集，返回101个子问题上的路径

def construct_routes_TSP50(shared_problem, pref):
    

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    # _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)


    
    pref = pref.cuda()

        
        
    # shared_problem = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])
    # test_path = f"./data/testdata_tsp_size{env_params['problem_size']}.pt"
    shared_problem = shared_problem.to(device=CUDA_DEVICE_NUM)
    
    test_num_episode = shared_problem.shape[0]
    tester_params['test_episodes'] = test_num_episode
    tester_params['test_batch_size'] = test_num_episode
    episode = 0
    
    sols = np.zeros([test_num_episode, 2])
    routes = np.zeros([test_num_episode, env_params['problem_size']])
    
        
    while episode < test_num_episode:

        tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
        
        remaining = test_num_episode - episode
        batch_size = min(tester_params['test_batch_size'], remaining)

        
        batch_sols, batch_routes = tester.run(shared_problem[episode: episode + batch_size],pref,batch_size)
        
        sols[episode:episode + batch_size] = batch_sols
        routes[episode:episode + batch_size] = batch_routes
        
        episode += batch_size
        
        # current_time = datetime.now()
        # formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        # print('now:{}, the episode now:{}, episodes left:{} '.format(formatted_time,episode,test_num_episode - episode))
        

        ############################
        # Logs
        ############################
        
        # all_done = (episode == test_num_episode)
        # if all_done:
        #     print('all done.')
    
    

    return routes

