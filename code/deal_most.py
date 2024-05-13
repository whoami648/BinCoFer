'''
Descripttion: 针对idapython 进行批量的脚本处理  进程号：1635730
version: 
Author: zyx
Date: 2023-11-14 15:24:19
LastEditors: zyx
LastEditTime: 2024-04-08 22:45:16
'''
from tqdm import tqdm
import os
import time
import ntpath
import datetime
import shutil
import matplotlib.pyplot as plt
import matplotlib
plt.set_loglevel("info") 
from optparse import OptionParser
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
import networkx as nx
import logging
import os
import sys
import re
import string
import random
import hashlib
import ntpath
import itertools
from hashlib import sha1
from subprocess import Popen, PIPE
from statistics import mean as stat_mean

import multiprocessing
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from contextlib import closing

ida_path = r"python"

script_path = r"/disk/zyx/test_base_feature/xiaorongshiyan/TPL_detection.py"

work_dir = r"/disk/zyx/test_base_feature/xiaorongshiyan/quanzhongwei1"

log_dir = r"/disk/zyx/test_base_feature/xiaorongshiyan/MI_save_path1"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#save_time = r"D:\Bincorer\Bincorer\time"
logger = logging.getLogger(__name__)

def run_helper(target):
    filename = target.split('\\')[-1]
    #cmd = '{} {} {} > {} 2>&1'.format(ida_path, script_path, os.path.join(work_dir, target),os.path.join(log_dir,filename+".log"))
    #print(cmd)
    cmd = [ida_path, script_path, os.path.join(work_dir, target)]
    with open(os.path.join(log_dir,filename+".log"), 'w') as logfile:
        ret = run(cmd, env=os.environ.copy(), stdout=logfile, stderr=logfile).returncode
        #subprocess.run(['python', '/path/to/your_script.py', file_path], stdout=logfile, stderr=logfile)
    
    print(cmd,"run finish")
    if ret != 0:
        logger.error("IDA returned {} for {}".format(ret, target))
        return target, False
    else:
        return target, True


def use_jiang_script():
    pool_size = int(7)
    print("pool_size",pool_size)
    target_list = os.listdir(work_dir)
    threshold = 1
    chunk_size = 3
    
    if len(target_list)>threshold:
        with closing(
                Pool(processes=pool_size)
        ) as pool:
            with tqdm(total=len(target_list)) as pbar:  # 添加总进度条
                data = []
                for result in pool.imap_unordered(run_helper, target_list, chunk_size):
                    data.append(result)
                    pbar.update(1)  # 更新总进度条
                    print()
            #data =list(pool.imap_unordered(run_helper, target_list, chunk_size))
    else:
        logger.debug("[+] no need to do multiprocessing because data is small.")
        data = []
        with tqdm(total=len(target_list)) as pbar: 
            for idx, arg in enumerate(target_list):
                data.append(run_helper(arg))
    return data
if __name__ == "__main__":
    use_jiang_script()