# -- coding: utf-8 --
# import idc
import time

import idautils
# import idaapi
import csv
import os
import pickle
import shutil
from collections import defaultdict
import idaapi,idc
from idaapi import *
from idautils import *
from idc import *
#import binaryai
import networkx as nx
#from utils import *
import sys
sys.setrecursionlimit(1000000)
# 使用方法：
# 使用run script批量运行本python文件，用于将传入的二进制程序通过ida提取相应的反编译信息，用于传入jtrans模型得到embedding
# func, asm_list, rawbytes_list, cfg, bai_feature

def get_asm(func):
    instGenerator = idautils.FuncItems(func)
    asm_list = []
    for inst in instGenerator:
        asm_list.append(idc.GetDisasm(inst))
    return asm_list

def get_cfg(func):

    def get_attr(block):
        asm,raw=[],b""
        curr_addr = block.start_ea
        while curr_addr < block.end_ea:
            asm.append(idc.GetDisasm(curr_addr))
            raw+=idc.get_bytes(curr_addr, idc.get_item_size(curr_addr))
            curr_addr = idc.next_head(curr_addr, block.end_ea)
        return asm, raw

    nx_graph = nx.DiGraph()
    flowchart = idaapi.FlowChart(idaapi.get_func(func), flags=idaapi.FC_PREDS)
    for block in flowchart:
        # Make sure all nodes are added (including edge-less nodes)
        attr = get_attr(block)
        nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])

        for pred in block.preds():
            nx_graph.add_edge(pred.start_ea, block.start_ea)
        for succ in block.succs():
            nx_graph.add_edge(block.start_ea, succ.start_ea)
    return nx_graph 

def get_rawbytes(func):
    instGenerator = idautils.FuncItems(func)
    rawbytes_list = b""
    for inst in instGenerator:
        rawbytes_list += idc.get_bytes(inst, idc.get_item_size(inst))
    return rawbytes_list

def get_func_pseudocode(ea):
    """
    get function pseudocode by IDA Pro
    Args:
        ea(ea_t): function address
    Returns:
        pseudocode(string): function pseudocode
    """
    try:
        hf = hexrays_failure_t()
        if IDA_SDK_VERSION >= 730:
            cfunc = decompile(ea, hf, DECOMP_NO_WAIT)
        else:
            cfunc = decompile(ea, hf)
        return str(cfunc)
    except Exception as e:
        print(str(e))
        return None
def get_binai_feature(func):
    return get_func_pseudocode(func)

def get_jTrains_data(saved_dict,saved_path):
    '''
    运行.py文件时会传入 被处理的单个binary的路径，处理存放到saved_path路径下
    应该过滤'.plt', 'extern', '.init', '.fini'段后生成jtrans的预处理数据
    :param saved_dict:
    :param saved_path:
    :return:
    '''
    count = 0
    # 记录时间
    start_time = time.time()

    print("======================== jTrans process start ========================")
    with open(saved_path, 'wb') as f:
        for func in idautils.Functions():
            if idc.get_segm_name(func) in ['.plt', 'extern', '.init', '.fini']:
                continue
            count += 1
            func_name = idc.get_func_name(func)
            asm_list = get_asm(func)  # 得到整个函数的汇编
            rawbytes_list = get_rawbytes(func)  # 得到二进制bit
            cfg = get_cfg(func)
            bai_feature = get_binai_feature(func)
            saved_dict[func_name] = [func, asm_list, rawbytes_list, cfg, bai_feature]
            print("count = ", count, "function ", func_name, "finish")
        total_time = time.time()-start_time
        binary_abs_path = get_input_file_path()
        filename = binary_abs_path.split('\\')[-1].split('/')[-1]
        with open(r"D:\Bincorer\Bincorer\jtrans_extract_time.csv","a+",newline="") as fd:
            writer = csv.writer(fd)
            writer.writerow([filename,total_time])

        #csvwriter = csv.writer(open(r"D:\OneDrive\TPL\BinCoFer\paper\time_logs\nix_new_preprocess_time_log.csv","a+"))
        #csvwriter.writerow(write_row)
        #pickle.dump(dict(saved_dict), f)
    print("======================== jTrans process finish ========================")





def preprocess_data(is_binary = False,BASEROOT = r""):

    #jtrans_pickle_dir = "jtrans_pickle_preprocessed_filtered"
    
    binary_abs_path = get_input_file_path()
    filename = binary_abs_path.split('\\')[-1].split('/')[-1]
    JTrans_SAVEROOT = BASEROOT
    jTrans_saved_path = os.path.join(JTrans_SAVEROOT, filename + "_extract_jTrains.pkl")  # unpair data


    saved_dict = defaultdict(lambda: list)

    get_jTrains_data(saved_dict, jTrans_saved_path)





if __name__ == '__main__':
    # process_jTrans()
    idaapi.autoWait()
    # extract_Modx_and_jTrans_data()
    # base_root = r"D:\data_for_tpl\BinCoFer\processed_data\tpl_so_new_102"
    # base_root = r"D:\data_for_tpl\BinCoFer\processed_data\nix_binary_strip_107"
    # base_root = r"D:\data_for_tpl\BinCoFer\processed_data\new_nix_binary"
    base_root = r"D:\Bincorer\Bincorer\nix_data\ans"
    if not os.path.exists(base_root):
        os.makedirs(base_root)
    # base_root = r"D:\data_for_tpl\BinCoFer\processed_data\ubuntu_binary_7\stripped"
    preprocess_data(BASEROOT=base_root)
    idc.qexit(0) # exit IDA







