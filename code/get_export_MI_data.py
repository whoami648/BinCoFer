'''
Descripttion: 针对MI,CC,LOC进行提取。
version: 1.0
Author: zyx
Date: 2023-11-14 15:13:11
LastEditors: zyx
LastEditTime: 2024-04-29 10:10:20
'''
import csv
import math
import os
import pickle
import time
from idaapi import *
from idautils import *
from idc import *
import idautils
import idc
import idaapi
from tqdm import tqdm
import networkx as nx
#from ida_process_data import get_cfg,get_func_pseudocode

BASE_PATH = r"D:\Bincorer\Bincorer\nix_data"

save_dir = os.path.join(BASE_PATH,r"MI_SCORE_fortimetest")#保存MI

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def save_pickle(file_name,file_object):
    '''
    name: 保存pickle
    msg: 
    param {*} file_name
    param {*} file_object
    return 
    '''
    with open(file_name,"wb") as f:
        pickle.dump(file_object,f)
    
def get_cfg( func):
    def get_attr(block, func_addr_set):
        asm, raw = [], b""
        curr_addr = block.start_ea
        if curr_addr not in func_addr_set:
            return -1
        # print(f"[*] cur: {hex(curr_addr)}, block_end: {hex(block.end_ea)}")
        while curr_addr <= block.end_ea:
            asm.append(GetDisasm(curr_addr))
            raw += get_bytes(curr_addr, get_item_size(curr_addr))
            curr_addr = next_head(curr_addr, block.end_ea)
        return asm, raw

    nx_graph = nx.DiGraph()
    flowchart = FlowChart(get_func(func), flags=FC_PREDS)
    func_addr_set = set([addr for addr in FuncItems(func)])
    for block in flowchart:
        # Make sure all nodes are added (including edge-less nodes)
        attr = get_attr(block, func_addr_set)
        if attr == -1:
            continue
        nx_graph.add_node(block.start_ea, asm=attr[0], raw=attr[1])
        # print(f"[*] bb: {hex(block.start_ea)}, asm: {attr[0]}")
        for pred in block.preds():
            if pred.start_ea not in func_addr_set:
                continue
            nx_graph.add_edge(pred.start_ea, block.start_ea)
        for succ in block.succs():
            if succ.start_ea not in func_addr_set:
                continue
            nx_graph.add_edge(block.start_ea, succ.start_ea)
    return nx_graph

def calculate_MI(func_ea):
    '''
    CC：圈复杂度
    V(G) = E - N + 2，E表示控制流图中边的数量，N表示控制流图中节点的数量
    HV = Nlog2(n) 表示词汇复杂性
    n = n1 + n2
    n1：程序中不同的操作符个数;    n2：程序中不同的操作数个数
    N1：程序中出现的操作符总数;    N2：程序中出现的操作数总数
    MI = 171 - 5.2 *ln(HV) - 0.23*(CC) - 16.2*ln(LOC))
    :return:MI,CC,Halstead Volume,Length of code
    '''
    cfg = get_cfg(func_ea)
    E = cfg.number_of_edges()
    N = cfg.number_of_nodes()
    CC = E-N+2
    # 传入function 首地址，计算复杂度; 获取到的asm会忽略掉db这种为指令
    instGenerator = idautils.FuncItems(func_ea)#获取函数所有的地址
    asm_list = []
    operator_N1_list = [] # 存放操作符
    operand_N2_list = [] # 存放操作数
    LOC = 0
    for inst in instGenerator:
        LOC+=1 #
        asm_list.append(idc.GetDisasm(inst))
        operatorstr = print_insn_mnem(inst)# 获得operator, 考虑在内的只保留后面接2个元素的操作符
        operand_attr0 = idc.get_operand_value(inst, 0)  # 得到操作数0
        operand_attr1 = idc.get_operand_value(inst, 1)  # 得到操作数1 ，返回的是数字
        operator_N1_list.append(operatorstr)
        operand_N2_list.extend([operand_attr0,operand_attr1])
    N1 = len(operator_N1_list)
    N2 = len(operand_N2_list)
    n1 = len(set(operator_N1_list))
    n2 = len(set(operand_N2_list))
    HV = (N1+N2)*math.log2(n1+n2)
    MI = 171 - 5.2 * math.log(HV) - 0.23 * (CC) - 16.2 * math.log(LOC)
    print(idc.GetFunctionName(func_ea),"MI",MI,"CC",CC,"Halstead Volume",HV,"LOC",LOC)
    return MI,CC,HV,LOC

def filter_supporting_function():
    '''
    用ida遍历一个程序，计算每个function的MI，存到pickle文件中
    使用复杂度计算公式筛选掉复杂度较低的function
    :param func_ea:
    :return:
    '''
    start_time = time.time()
    MI_dict = {}
    for func_ea in idautils.Functions():
        MI,CC,HV,LOC = calculate_MI(func_ea)
        func_name = idc.GetFunctionName(func_ea)
        MI_dict[func_name] = {"MI":MI,"CC":CC,"HV":HV,"LOC":LOC}
    
    # 写入的文件名称
    # binary_abs_path = get_input_file_path()
    # filename = binary_abs_path.split('\\')[-1].split('/')[-1]+"_MI.pkl"

    # dump_pickle_filename = os.path.join(save_dir,filename)
    # with open(dump_pickle_filename,"wb") as f:
    #     pickle.dump(MI_dict,f)
    #print("write MI score to ",dump_pickle_filename," finished")
    return time.time()-start_time
    

def get_export_function_list():
    '''
    name: get_export_function_list
    msg: 获取export表中的函数
    return {*}
    '''
    start = time.time()
    export_functions = idautils.Entries()  # return: List of tuples (index, ordinal, ea, name)
    module_function_names = set()
    for (index, ordinal, ea, ori_func_name) in tqdm(export_functions):
        # 识别一下ea在.text段，是函数不是数据
        if idc.SegName(ea) != ".text":
            continue
        func_name = idc.GetFunctionName(ea)  # 由于Entries中获得的函数名和function中得到的函数名不一致，因此重新使用idc获得函数名
        if func_name == "":
            if ori_func_name != "":
                func_name = ori_func_name
            else:
                continue
        module_function_names.add(func_name)

    save_path = r"D:\Bincorer\Bincorer\nix_data\export"

    binary_abs_path = get_input_file_path()
    filename1 = binary_abs_path.split('\\')[-1].split('/')[-1]
    file_name = os.path.join(save_path,filename1+"_export.pkl")
    
    #save_pickle(file_name,list(module_function_names))
    print(file_name + "export save over") 
    return start-time.time()

    return list(module_function_names)
if __name__=="__main__":
    idaapi.autoWait()






    save_path = r"D:\Bincorer\Bincorer\save_path"
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    MI_time = filter_supporting_function()
    export_time = get_export_function_list()


    binary_abs_path = get_input_file_path()
    filename1 = binary_abs_path.split('\\')[-1].split('/')[-1]
    file_name = os.path.join(save_path,r"archlinux_time.csv")#filename1+

    with open(file_name,"a+",newline="") as f:
        writer = csv.writer(f)
        writer.writerow([filename1,MI_time,export_time])
    



    idc.qexit(0) # exit IDA