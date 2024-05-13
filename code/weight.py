import pickle
import torch
import os
import csv
from tqdm import tqdm
import os

# 使用第二张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
#export_path = r"/disk/zyx/test_base_feature/export"
cos_sim_threshold_list = [0.8]
#全局变量
work_dir = r"xiaorongshiyan/MI_save_path" #Mi_path
save_path2 = r"xiaorongshiyan/MI_save_path1"#过滤之后的最终结果
#创建最终文件

if not os.path.exists(save_path2):
    os.makedirs(save_path2)

#开始
def write_embedding(embedding,filename):
    '''写入pickle文件，{embedding:[1,2,3],..},filename = '''
    with open(filename, 'wb') as f:
        pickle.dump(embedding, f)
    print("save to",filename," finish.")

def load_pickle_file(save_filename):
    '''load pickle'''
    with open(save_filename, 'rb') as fr:
        try:
            obj = pickle.load(fr)
        except UnicodeDecodeError: # python3
            fr.seek(0)
            obj = pickle.load(fr, encoding = 'latin1')
    return obj

def read_a_csv(filename):
    '''file_object[funcname]=func_emb_list,func_list,file_func_matrix'''
    file_object = {}
    file_func_matrix = []
    func_list = []
    '''export filter'''
    #export_name = os.path.basename(filename).replace(".csv","_export.pkl")
    #export_list = load_pickle_file(os.path.join(export_path,export_name))
    with open(filename,"r") as csvfiles:
        reader = csv.reader(csvfiles)
        for row in reader:
            if row == None:
                continue
            funcname = row[0]#os.path.basename(filename).replace(".csv","")+"_"+
            # if funcname not in export_list:
            #     continue
            func_emb = row[1]
            # turn func embed to a list
            # for num in func_emb.split(","):
            #     n = num.strip(" ").strip("[").strip("]")
            func_emb_list = [float(num.strip().strip("[").strip("]")) for num in func_emb.split(",")]
            #print(row[1])
            #print(func_emb_list)
            file_object[funcname]=torch.tensor(func_emb_list).to(device)
            func_list.append(funcname)
            file_func_matrix.append(func_emb_list)
    return file_object,func_list,torch.tensor(file_func_matrix).to(device)

def append_to_csvfile(filename,csvrow):
    '''按行存储'''
    with open(filename,"a+",newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csvrow)
        #csvfile.flush()


def load_data(TPL_repo_path):
    #加入export过滤的
    print("=============================================start to load TPL datas=============================================")
    all_data = {}
    cnt = 0
    for tpl in os.listdir(TPL_repo_path):
        if "ld-linux-x86-64.so.2" in tpl:
            continue
        file_objet,func_name,func_matrix = read_a_csv(os.path.join(TPL_repo_path,tpl))

        tpl_name = tpl.replace(".csv","")
        # if_else = {}
        all_data[tpl_name] = {}
        all_data[tpl_name]["func_name"] = func_name
        all_data[tpl_name]["func_matrix"] = func_matrix

        cnt+=len(func_name)

    print("loading datas over")
    print("data number:"+str(cnt))
    return all_data,cnt

def cos_calculate_functions(save_path2):

    ''''
    input : save_path
    return : 权重结果，并且保存到save——path
    '''
    for MI in ["MI_0.2"]:
        TPL_repo_path  = work_dir
        all_data,cnt = load_data(TPL_repo_path)
        print("=============================================loading {MI} TPL datas over=============================================")
        print()
        print("=========================================start to cos_calculate_functions========================================")
        
        all_tpl_name = list(all_data.keys())

        for cos_sim_threshold in cos_sim_threshold_list:#阈值

            save_path3 = os.path.join(save_path2,MI+"_sim_"+str(cos_sim_threshold))
            if not os.path.exists(save_path3):
                os.mkdir(save_path3)
            for i in tqdm(range(len(all_tpl_name))):#当前TPL
                
                input_name = all_tpl_name[i]
                input_func = all_data[input_name]["func_name"]
                input_func_embedding = all_data[input_name]["func_matrix"]
                df_result = torch.tensor([0]*len(input_func)).to(device) #创建所有的函数最终df结果

                print("=========================================================start to calculate "+input_name+" with others=========================================================")
                print("len(input_func) : ",len(input_func))
                if len(input_func_embedding)==0:
                    continue
                df = torch.tensor([1] *len(input_func)).to(device)



                for j in range(len(all_tpl_name)):
                    #读取所有比较值
                    compared_name = all_tpl_name[j] 
                    compared_func = all_data[compared_name]["func_name"]
                    compared_func_embedding = all_data[compared_name]["func_matrix"]
                    if len(compared_func_embedding.t())==0:
                        continue
                    if i == j: #自身对比查找，tf = 函数在本TPL中出现的数量 / 当前TPL的函数数量

                        the_dot_result = torch.mm(input_func_embedding, compared_func_embedding.t())
                        input_func_embedding_norm = torch.norm(input_func_embedding, p=2, dim=1, keepdim=True)  # (input_batch size,1)
                        compared_func_embedding_norm = torch.norm(compared_func_embedding.t(), p=2, dim=0,keepdim=True)  # (1,batch size32)
                        the_total_norm = torch.mm(input_func_embedding_norm, compared_func_embedding_norm)
                        cosine_result = torch.div(the_dot_result, the_total_norm)  # (input batch size, compare batch size)

                        df_result = torch.sum(cosine_result > cos_sim_threshold, dim=1) #判断每一行大于cos_sim_threshold的数量
                        TF = df_result / len(compared_func) #计算Tf矩阵
                        continue

                    #计算两个文件之间的cos值
                    the_dot_result = torch.mm(input_func_embedding, compared_func_embedding.t())
                    input_func_embedding_norm = torch.norm(input_func_embedding, p=2, dim=1, keepdim=True)  # (input_batch size,1)
                    compared_func_embedding_norm = torch.norm(compared_func_embedding.t(), p=2, dim=0,keepdim=True)  # (1,batch size32)
                    the_total_norm = torch.mm(input_func_embedding_norm, compared_func_embedding_norm)
                    cosine_result = torch.div(the_dot_result, the_total_norm)  # (input batch size, compare batch size)

                    df_result = torch.any(cosine_result > cos_sim_threshold,dim=1).int() #判断每一行是否含有大于cos_sim_threshold
                    df+=df_result #计算df的值
                IDF = (len(all_tpl_name)/df).log()#计算IDF

                TF_IDF = TF*IDF #计算出函数对应的值
                all_TF_IDF = torch.sum(TF_IDF).item()

                
                for i in range(len(input_func)):
                    df = TF_IDF[i].item()/all_TF_IDF#当前input文件函数的最终结果
                    append_to_csvfile(os.path.join(save_path3,input_name+".csv"),[input_func[i],df,input_func_embedding[i].tolist()])
                #all_func_91.append([input_name,input_func[i],df])

                #all_func = sorted(all_func_91,key=lambda x:x[2])[:int(cnt*df_percent)]#按百分比过滤后的结果
                #for i in all_func:
                    #append_to_csvfile(os.path.join(save_path,"all_func.csv"),i)
                    #append_to_csvfile(os.path.join(save_path3,i[0]+"_df_filter.csv"),(i[1],i[2]))#（最终的结果）
                #print("sim_"+str(cos_sim_threshold)+"_df_"+str(df_percent)+" over")
            torch.cuda.empty_cache()

            print(input_name,"calculate over")
            
        torch.cuda.empty_cache()
    return cnt

if __name__=='__main__':

  
    cos_calculate_functions(save_path2)


   