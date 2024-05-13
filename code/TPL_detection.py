import csv
import torch
import os
import time
#sct = 0.91
import sys 
import json 
sys.setrecursionlimit(3000)
from tqdm import tqdm
import argparse
#当前进程号

# 设备配置
# 使用第二张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cpu') 


#读取基本路径
# parser = argparse.ArgumentParser(description='Process files in a folder.')
# parser.add_argument('folder_path', type=str, help='Path to the folder to process')
# args = parser.parse_args()
# folder_path = args.folder_path

#基本路径
TPL_repo_path = r"archlinux_new_TPL_detection/MI_export_df_filter_dl/zuiyouMI_0.2_sim_0.8"#folder_path#
binary_path = r"/disk/zyx/test_base_feature/xiaorongshiyan/binary"
cos_sim_threshold_list = [0.8]
save_path = r"xiaorongshiyan/ans_myquanzhong"
save_time_path = r"save_time_path"
if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(save_time_path):
    os.mkdir(save_time_path)

'''ground_truth = {
                'tcpdump': {'libpcap', 'libcap-ng', 'libc-2.27'}, 
                'openssl': {'libdl-2.27', 'libc-2.27', 'libcrypto', 'libssl', 'libpthread-2.27'}, 
                'vim': {'libdl-2.27', 'libm-2.27', 'libtinfo', 'libc-2.27'}, 
                'busybox': {'libm-2.27', 'libresolv-2.27', 'libc-2.27'}, 
                'openvpn': {'libc-2.27', 'libcrypto', 'libssl', 'libpthread-2.27', 'liblzo2'}, 
                'sqlite3': {'libdl-2.27', 'libpthread-2.27', 'libc-2.27'}, 
                'ssldump': {'libpcap', 'libc-2.27'}
                }'''
#'git': {'libz', 'libpthread-2.27', 'libc-2.27'}, 'watcher': {'libpcap'},
def append_to_json(filename,item):
    with open(filename,"w+") as f:
        json.dump(item,f)
def append_to_csvfile(filename,csvrow):
    '''按行存储'''
    with open(filename,"a+",newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvrow)
        #csvfile.flush()
def read_TPL_csv(filename):
    '''file_object[funcname]=func_emb_list,func_list,file_func_matrix'''
    file_object = {}
    file_func_matrix = []
    func_list = []
    func_weight = []
    with open(filename,"r") as csvfiles:
        reader = csv.reader(csvfiles)
        for row in reader:
            if row == None:
                continue
            funcname = row[0]#os.path.basename(filename).replace(".csv","")+"_"+
            func_emb = row[2]
            weight = eval(row[1])
            # turn func embed to a list
            # for num in func_emb.split(","):
            #     n = num.strip(" ").strip("[").strip("]")
            func_emb_list = [float(num.strip().strip("[").strip("]")) for num in func_emb.split(",")]
            #print(row[1])
            #print(func_emb_list)
            #file_object[funcname]=torch.tensor(func_emb_list).to(device)
            #func_list.append(funcname)
            file_func_matrix.append(func_emb_list)
    func_weight = [1/len(file_func_matrix)]*len(file_func_matrix)
    return torch.tensor(file_func_matrix).to(device),torch.tensor(func_weight).to(device)

def read_binary_csv(filename):
    '''file_object[funcname]=func_emb_list,func_list,file_func_matrix'''
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
            func_emb_list = [float(num.strip().strip("[").strip("]")) for num in func_emb.split(",")]
            #print(row[1])
            #print(func_emb_list)
            #file_object[funcname]=torch.tensor(func_emb_list).to(device)
            #func_list.append(funcname)
            file_func_matrix.append(func_emb_list)

    return torch.tensor(file_func_matrix).to(device)
def Find_candidate(df_result):
    '''根据计算出来的相似度返回TPL候选列表'''
    # 找到每一列的最大值
    max_per_column, _ = torch.max(df_result, dim=0)

    # 将每一列的最大值相加
    sum_of_max_values = torch.sum(max_per_column)
    return sum_of_max_values.item()

def caculate_similarity(input_func_embedding, compared_func_embedding,weight,sct):
    '''计算相似度,返回是否算'''
    if len(compared_func_embedding)==0:
        return torch.tensor([0]).to(device)
    the_dot_result = torch.mm(input_func_embedding, compared_func_embedding.t())
    #a = torch.cosine_similarity(input_func_embedding[0], compared_func_embedding[0],dim=0)
    input_func_embedding_norm = torch.norm(input_func_embedding, p=2, dim=1, keepdim=True)  # (input_batch size,1)
    compared_func_embedding_norm = torch.norm(compared_func_embedding.t(), p=2, dim=0,keepdim=True)  # (1,batch size32)
    the_total_norm = torch.mm(input_func_embedding_norm, compared_func_embedding_norm)
    cosine_result = torch.div(the_dot_result, the_total_norm)  # (input batch size, compare batch size)
    #df_result = torch.any(cosine_result > sct) #判断是否含有大于sct的值
    #cosine_result[cosine_result<sct] = 0
    df_result = cosine_result*weight
    return Find_candidate(df_result)

def jiance(ground_truth):
    solved = []
    # with open(r"archlinux_end_test/solved.csv","r") as f:
    #     reader = csv.reader(f)
    #     for read in reader:
    #         if read[0]=="sim":
    #             continue
    #         solved.append(read[0])

    #append_to_csvfile(os.path.join(save_path,"all_ans_archlinux.csv"),["sim","MI","Precision","Recall"])
    sim_df_MI = os.path.basename(TPL_repo_path)
    if sim_df_MI in solved:
        sys.exit()

    all_binary={}

    print("=========================================load all binary============================================")
    for binary in tqdm(os.listdir(binary_path)):
        if binary.replace(".csv","") in ground_truth.keys():
            all_binary[binary]=read_binary_csv(os.path.join(binary_path,binary))
    print("=========================================loading over============================================\n")

    print("=========================================start to cacluate sim============================================")
    

    
    #MI = eval(sim_df_MI.split("_")[1])
    sim = eval(sim_df_MI.split("_")[3])
    #MI = eval(sim_df_MI.split("_")[5])
    #a = [str(sim),str(df),str(MI)]

    TPL_repo_path1 = TPL_repo_path

    ans={}
    ans_time = {}
    for binary in ground_truth.keys():
        ans[binary]=[]
        binary_ans = []#当前bianry的检测结果
    for binary,input_func_embedding in all_binary.items():#导入binary张量
        #input_func_embedding = read_a_csv(os.path.join(binary_path,binary))#读取全部的binary张量
        
        times=time.time()
        binary_ans = []#当前bianry的检测结果

        #TPL    -+
        for tpl in os.listdir(TPL_repo_path1):
            compared_func_embedding,weight = read_TPL_csv(os.path.join(TPL_repo_path1,tpl))
            flag = caculate_similarity(input_func_embedding, compared_func_embedding,weight,sim)
            #if flag:#得到当前TPL的复用结果
            
            tpl_name = tpl.replace(".csv","")
            binary_ans.append((tpl_name,flag))
                #print([binary.replace(".csv",""),tpl_name])
        cost_time = time.time()-times
        ans_time[binary.replace(".csv","")] = cost_time
        ans[binary.replace(".csv","")] = sorted(binary_ans,key=lambda x:x[1],reverse=True)

    save_name = os.path.join(save_path,sim_df_MI+".json")
    append_to_json(save_name,ans)

        
        #for key,values in ans.items():
            #append_to_csvfile(sim_df_MI+".csv",[key,values])
        #Precision,Recall = Precision_Recall(file_name=os.path.join(save_path,sim_df_MI+".csv"),binary_similarity=ans,ans_time=ans_time)
        #append_to_csvfile(os.path.join(save_path,"all_ans_archlinux.csv"),[sim,df,MI,Precision,Recall])
        #print(sim,df,MI,Precision,Recall)
            
    return ans

def jiance3(ground_truth):
    solved = []
    # with open(r"archlinux_end_test/solved.csv","r") as f:
    #     reader = csv.reader(f)
    #     for read in reader:
    #         if read[0]=="sim":
    #             continue
    #         solved.append(read[0])
    save_path = r"archlinux_test_MI/ans"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    append_to_csvfile(os.path.join(save_path,"all_ans_archlinux.csv"),["sim","df","MI","Precision","Recall"])

    all_binary={}

    print("=========================================load all binary============================================")
    for binary in tqdm(os.listdir(binary_path)):
        if binary.replace(".csv","") in ground_truth.keys():
            all_binary[binary]=read_a_csv(os.path.join(binary_path,binary))
    print("=========================================loading over============================================\n")

    print("=========================================start to cacluate sim============================================")
    
    

    for sim_df_MI in tqdm(os.listdir(TPL_repo_path)):
        if sim_df_MI in solved:
            continue
        
        MI = eval(sim_df_MI.replace("MI_",""))
        #a = [str(sim),str(df),str(MI)]
        if MI>=0.1:
            continue
        

        TPL_repo_path1 = os.path.join(TPL_repo_path,sim_df_MI)

        for sim in cos_sim_threshold_list:
            ans={}
            ans_time = {}
            for binary in ground_truth.keys():
                ans[binary]=set()
            for binary,input_func_embedding in all_binary.items():#导入binary张量
                #input_func_embedding = read_a_csv(os.path.join(binary_path,binary))#读取全部的binary张量
                

            

                times=time.time()

                #TPL    -+
                for tpl in os.listdir(TPL_repo_path1):
                    compared_func_embedding = read_a_csv(os.path.join(TPL_repo_path1,tpl))

                
                    flag = caculate_similarity(input_func_embedding, compared_func_embedding,sim)
                    if flag:#得到当前TPL的复用结果
                        tpl_name = tpl.replace(".csv","")
                        ans[binary.replace(".csv","")].add(tpl_name)
                        #print([binary.replace(".csv",""),tpl_name])
                cost_time = time.time()-times
                ans_time[binary.replace(".csv","")] = cost_time
            #for key,values in ans.items():
                #append_to_csvfile(sim_df_MI+".csv",[key,values])
            Precision,Recall = Precision_Recall(file_name=os.path.join(save_path,"sim_"+str(sim)+"_"+sim_df_MI+".csv"),binary_similarity=ans,ans_time=ans_time)
            append_to_csvfile(os.path.join(save_path,"all_ans_archlinux.csv"),[sim,"None",MI,Precision,Recall])
            print(sim,MI,Precision,Recall)
            
    return ans
def TPL_Precision_Recall(value,value_list):
    '''value检测值,value_list真值'''
    tp = len(value.intersection(value_list)) #交集
    fp = len(value.difference(value_list))
    fn = len(value_list.difference(value))
    if tp+fp>0:
        Precision = tp/(tp+fp)
    else:
        Precision = "error"
    if tp+fn>0:
        Recall = tp/(tp+fn)
    else:
       Recall  = "error"

    return "Precision："+str(Precision)+" Recall："+str(Recall)


from time import gmtime
from time import strftime
def jiance1(binary):
    ans=set()
    binary = os.path.basename(binary)+".csv"
    input_func_embedding = read_a_csv(os.path.join(binary_path,binary))
    a=time.time()
    for tpl in os.listdir(TPL_repo_path):
        if "TPL_repo" in tpl:
            continue
        compared_func_embedding = read_a_csv(os.path.join(TPL_repo_path,tpl))
        flag = caculate_similarity(input_func_embedding, compared_func_embedding)
        if flag:
            tpl_name = tpl[:tpl.find(".so")]
            ans.add(tpl_name)
            #print(binary,tpl_name)
    return str(ans)+"\n"+TPL_Precision_Recall(ans,ground_truth[binary.replace(".csv","")])#+"匹配时间："+str((time.time()-a)*len(input_func_embedding))

def Precision_Recall(file_name,binary_similarity,ans_time):
    '''计算精确率，binary_similarity = {binary:{TPL1,TPL2}}'''
    #binary_similarity = jiance()#{'tcpdump': {'libpcap', 'libc-2.27'}, 'openssl': {'libssl', 'libpthread-2.27', 'libcrypto', 'libc-2.27'}, 'vim': {'libtinfo', 'libc-2.27'}, 'busybox': {'libc-2.27'}, 'openvpn': {'libpthread-2.27', 'libcrypto', 'liblzo2', 'libc-2.27'}, 'sqlite3': {'libpthread-2.27', 'libc-2.27'}, 'ssldump': {'libpcap', 'libc-2.27'}}
    #print(binary_similarity)
    TP,tp = {},0 #真正的正例
    FP,fp = {},0 #错误的正例
    FN,fn = {},0 #错误的反例
    #print(ans_time)

    ans = []
    
    for binary,value in binary_similarity.items():
        value_list = ground_truth[binary] #真实数据集
        TP[binary] = len(value.intersection(value_list)) #交集
        FP[binary] = len(value.difference(value_list))
        FN[binary] = len(value_list.difference(value))
        #print(TP[binary],FP[binary],FN[binary])
        if TP[binary]+FP[binary]>0:
            Precision = TP[binary]/(TP[binary]+FP[binary])
        else:
            Precision = "error"
        if TP[binary]+FN[binary]>0:
            Recall = TP[binary]/(TP[binary]+FN[binary])
        else:
            Recall = "error"
        ans.append((binary,TP[binary],FP[binary],FN[binary],"",Precision,Recall,"",value,"",0))
        tp+=TP[binary]
        fp+=FP[binary]
        fn+=FN[binary]
    #
    # print(value)
    #写入csv
    if tp+fp>0:
        Precision = tp/(tp+fp)
    else:
        Precision = "error"
    if tp+fn>0:
        Recall = tp/(tp+fn)
    else:
       Recall  = "error"
    
    header = ["binary","TP","FP","FN","","Precision","Recall","","value","","time"]
    with open(file_name,"a+",encoding='utf-8',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(ans)
        writer.writerow([])
        writer.writerow(("summarize",tp,fp,fn,"",Precision,Recall))

    return Precision,Recall

if __name__=="__main__":


    #file_name = "TPL_ans.csv"
    train_ground_truth = {}
    with open(r"archlinux_new_TPL_detection/archlinux_ground_truth_test1.csv","r") as f:
        reader = csv.reader(f)
        for read in reader:
            if read[0]=="binary":
                continue
            train_ground_truth[read[0]] = set(read[1:])#.discard('')#).discard('')#删除空元素，可能因为读取最后
            #print()
            train_ground_truth[read[0]].discard('')
    ground_truth = train_ground_truth
    a = time.time()
    jiance(ground_truth)
    
    #print(jiance())
    #Precision_Recall(file_name)
    #value,value_list = {'libdl-2.27', 'libpthread-2.27'},{'libdl-2.27', 'libpthread-2.27', 'libc-2.27'}
    #print(TPL_Precision_Recall(value,value_list))
    print("total time",time.time()-a)