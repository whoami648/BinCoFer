'''
Descripttion: 
version: V1.0
Author: zyx
Date: 2023-11-24 10:08:53
LastEditors: zyx
LastEditTime: 2023-12-27 20:32:57
'''

import os

# 使用第二张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle
import csv
import torch
import time
import sys  # ����sysģ��
sys.setrecursionlimit(3000)
from tqdm import tqdm

#需要过滤的阈值范围
df_saved_percentage = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.11,0.12,0.13,0.14,0.15, 0.35, 0.45,0.1, 0.2, 0.3, 0.4, 0.5]
cos_sim_threshold_list = [0.6, 0.7, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
MI_filered_percentage_list = [0.2]#[0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7]

#初始路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备

export_path = r"xiaorongshiyan/export"#exports\libacl.so.1.1.0.pkl
df_path = r"df_ans"#df_ans\sim_0.5_df_0.1\libacl.so.1.1.0_df_filter.csv
MI_path = r"xiaorongshiyan/MI_SCORE_fortimetest"#MI_SCORE_fortimetest\libacl.so.1.1.0_MI.pkl
TPL_function_embedding_path = r"xiaorongshiyan/TPL"#libasn1.so.8.0.0.csv
binary_function_embedding_path = r"xiaorongshiyan/binary"

save_path = r"archlinux_new_TPL_detection/export_TPL_repo"#保存路径


#创建最终文件
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(df_path):
    os.mkdir(df_path)


#读取
def read_a_csv(filename):
    '''file_object[funcname]=func_emb_list'''
    file_object = {}
    file_func_matrix = []
    func_list = []
    with open(filename,"r") as csvfiles:
        reader = csv.reader(csvfiles)
        for row in reader:
            if row == None:
                continue
            funcname = row[0]#os.path.basename(filename).replace(".csv","")+"_"+
            func_emb = row[1]
            # turn func embed to a list
            # for num in func_emb.split(","):
            #     n = num.strip(" ").strip("[").strip("]")
            #func_emb_list = [float(num.strip().strip("[").strip("]")) for num in func_emb.split(",")]
            #print(row[1])
            #print(func_emb_list)
            file_object[funcname]=row[1]#func_emb_list#torch.tensor(func_emb_list).to(device)#func_emb_list#torch.tensor(func_emb_list).to(device)
            #func_list.append(funcname)
            #file_func_matrix.append(func_emb_list)
    return file_object

def load_pickle_file(save_filename):
    '''读取pickle文件'''
    with open(save_filename, 'rb') as fr:
        try:
            obj = pickle.load(fr)
        except UnicodeDecodeError: # python3
            fr.seek(0)
            obj = pickle.load(fr, encoding = 'latin1')
    return obj

#保存csv和pickle
def append_to_csvfile(filename,csvrow):
    '''
    将csvrow追加到filename的csv格式文件中
    :param filename:
    :param csvrow:
    :return:
    '''
    with open(filename,"a+",newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvrow)
        csvfile.flush()

def save_pickle_file(obj, save_filename):
    with open(save_filename, 'wb') as fw:
        pickle.dump(obj, fw)
    print ("Object with type '%s' has saved to file '%s'." % (type(obj), save_filename))

def load_TPL_datas():
    '''
    description: 针对所有的TPL数据进行导入
    return export:{name:export_list}
           MI:{file_object { funcname:MI,.. },.. }#已经排序
           TPL:{file_object { funcname:func_emb_list,.. },.. }
    '''   
    print("=============================================start to load TPL datas=============================================") 
    export = {}
    for i in os.listdir(export_path):
        export_name = i.replace("_export.pkl","")
        if "libdl" in i or "ld" in i or "libthpthread" in i:
            continue
        export_list = load_pickle_file(os.path.join(export_path,i))
        export[export_name]=export_list

    #返回排序后的MI
    MI = {}
    for i in os.listdir(MI_path):
        if "libdl" in i or "ld" in i or "libthpthread" in i:
            continue
        MI_name = i.replace("_MI.pkl","")
        MI_list1 = load_pickle_file(os.path.join(MI_path,i))
        MI_list = dict(sorted(MI_list1.items(),key=lambda a:a[1]["MI"]))
        MI[MI_name]=list(MI_list.keys())
    
    print("=============================================loading TPL datas over=============================================")
    print()

    return export,MI
    
def load_TPL_function_embedding():
    TPL_function_embedding = {}
    for i in os.listdir(TPL_function_embedding_path):
        tpl_name = i.replace(".csv","")
        func_embedding = read_a_csv(os.path.join(TPL_function_embedding_path,i))
        TPL_function_embedding[tpl_name] = func_embedding

    print("=============================================loading TPL datas over=============================================")
    print()

    return TPL_function_embedding
def load_binary_datas():

    cnt = 0


    print("=============================================start to load binary datas=============================================") 
    binary_function_embedding = {}
    for i in os.listdir(binary_function_embedding_path):
        tpl_name = i.replace(".csv","")
        func_embedding = read_a_csv(os.path.join(binary_function_embedding_path,i))
        binary_function_embedding[tpl_name] = func_embedding
        cnt+=len(func_embedding)
    print("binary_len：",len(binary_function_embedding),"bianry_fun: ",cnt)
    print("=============================================loading binary datas over=============================================")
    print()
    return binary_function_embedding

def Create_TPL_repo():
    export,MI = load_TPL_datas()
    TPL_repo_path = save_path
    if not os.path.exists(TPL_repo_path):
        os.makedirs(TPL_repo_path)

    #MI过滤
    for MI_filter in MI_filered_percentage_list:
        start_time = time.time()
        print("=============================================MI_filter："+str(MI_filter)+"=============================================")
        for key,value in tqdm(MI.items()):
            
            MI_values =set(value[:int(len(value)*MI_filter)])  #d当前MI
            export_values = set(export[key])
            embedding_name = key+".csv"

            TPL_embedding = read_a_csv(os.path.join(TPL_function_embedding_path,embedding_name))
            print("=============================================="+key+"==============================================")


            for sim_df in os.listdir(df_path):
                name = key+"_df.csv"
                df_values = set()
                if os.path.exists(os.path.join(df_path,sim_df,name)): #检查是否存在
                    with open(os.path.join(df_path,sim_df,name),"r") as f:
                        reader = csv.reader(f)
                        for read in reader:
                            df_values.add(read[0])

                save_path1 = os.path.join(TPL_repo_path,sim_df+"_MI_"+str(MI_filter))
                if not os.path.exists(save_path1):
                    os.mkdir(save_path1)

                #取交集
                #ans = MI_values.intersection(export_values)
                ans = list(MI_values.intersection(df_values))
                #print(sim_df,ans)
                if len(ans)==0:
                    continue
            #     with open(os.path.join(save_path1,key+".csv"),"w+")as f:
            #         writer = csv.writer(f)
            #         for i in ans:
            #             writer.writerow([i,TPL_embedding[i]])
            # print(key+".csv "+" over")
        print(time.time()-start_time)


def Create_TPL_repo1():
    export,MI = load_TPL_datas()
    TPL_repo_path = save_path
    if not os.path.exists(TPL_repo_path):
        os.makedirs(TPL_repo_path)
    MI_cnt = 0
    export_cnt = 0
    cnt = 0

    #MI过滤
    for MI_filter in MI_filered_percentage_list:
        print("=============================================MI_filter："+str(MI_filter)+"=============================================")
        start_time = time.time()
        for key,value in tqdm(MI.items()):

            export_values = set(export[key])
            value1=list(set(value).intersection(export_values))#value#
            length = int(len(value1)*MI_filter)
            if length<=1:
                length = 2
            MI_values =set(value1[:length])  #d当前MI
            embedding_name = key+".csv"
            if embedding_name not in os.listdir(TPL_function_embedding_path):
                continue

            #TPL_embedding = read_a_csv(os.path.join(TPL_function_embedding_path,embedding_name))
            print("=============================================="+key+"==============================================")




                #取交集
                #ans = MI_values.intersection(export_values)
            ans = list(MI_values)#.intersection(export_values))
            #print(sim_df,ans)
            save_path1 = os.path.join(save_path,"MI_"+str(MI_filter))
        print(time.time()-start_time)
        break
            # if not os.path.exists(save_path1):
            #     os.mkdir(save_path1)
            # with open(os.path.join(save_path1,key+".csv"),"w+")as f:
            #     writer = csv.writer(f)
            #     for i in ans:
            #         if i in TPL_embedding.keys():
            #             writer.writerow([i,TPL_embedding[i]])
            #             MI_cnt+=1
            # print(key+".csv "+" over")

            #export
            # print("=================================================export start========================================")

            # save_path1 = r"export_save_path"
            # if not os.path.exists(save_path1):
            #     os.mkdir(save_path1)
            # with open(os.path.join(save_path1,key+".csv"),"w+")as f:
            #     writer = csv.writer(f)
            #     for i in export[key]:
            #         cnt += len(TPL_embedding.keys())
            #         if i in TPL_embedding.keys():
            #             writer.writerow([i,TPL_embedding[i]])
            #             export_cnt +=1
            # print(key+".csv "+" over")
  
    # print("MI_cnt：",MI_cnt)
    # print("export_cnt：",export_cnt)
    # print("cnt：",cnt)



        


if __name__=="__main__":


    # 暂存，用于恢复
    
    Create_TPL_repo1()
    #export,MI= load_TPL_datas()
    #load_binary_datas()
