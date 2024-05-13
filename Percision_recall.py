import csv
import torch
import os
import time
#sct = 0.91
import sys 
import json 
sys.setrecursionlimit(3000)
from tqdm import tqdm
def append_to_csvfile(filename,csvrow):
    '''按行存储'''
    with open(filename,"a+",newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvrow)

def Precision_Recall(file_name,binary_similarity,ground_truth):
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
        ans.append((binary,TP[binary],FP[binary],FN[binary],"",Precision,Recall,"",value))
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
    
    header = ["binary","TP","FP","FN","","Precision","Recall","","value"]
    with open(file_name,"a+",encoding='utf-8',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(ans)
        writer.writerow([])
        writer.writerow(("summarize",tp,fp,fn,"",Precision,Recall))

    return Precision,Recall

if __name__=="__main__":

    topk = 5
    save_path = r"xiaorongshiyan/pericision_recall5"
    cos_thresold_all = [0.8]
    #jiance_jieguo_path = 
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    
    #file_name = "TPL_ans.csv"
    append_to_csvfile(os.path.join(save_path,r"all_ans_test.csv"),["sim","MI","sim2","Precision","Recall","F1"])
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
    for cos_thresold in tqdm(cos_thresold_all):
        for i in os.listdir(r"xiaorongshiyan/ans_myquanzhong"):
            file = os.path.join(r"xiaorongshiyan/ans_myquanzhong",i)
            MI = i.replace(".json","").split("_")[1]
            sim = i.replace(".json","").split("_")[3]
            with open(file,"r") as f:
                ans = json.load(f)
            binary_similarity = {}
            for key,values in ans.items():
                binary_similarity[key] = set()
                cnt = 0
                for j in range(len(values)):
                    if "libdl" in values[j][0] or "libpthread" in values[j][0]:
                        continue
                    if values[j][1]>cos_thresold:
                        binary_similarity[key].add(values[j][0])
                    # cnt+=1
                    # if cnt>=len(ground_truth[key]):
                    #     break
                    
            file_name = os.path.join(save_path,i.replace(".json","sim2_"+str(cos_thresold)+".csv"))
            Precision,Recall = Precision_Recall(file_name,binary_similarity,ground_truth)
            append_to_csvfile(os.path.join(save_path,r"all_ans_test1.csv"),[sim,MI,cos_thresold,Precision,Recall,2*Precision*Recall/(Precision+Recall)])
        