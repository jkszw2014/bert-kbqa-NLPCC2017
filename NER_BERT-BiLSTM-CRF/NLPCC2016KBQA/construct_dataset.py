# coding:utf-8
import sys
import os
import pandas as pd


'''
构造NER训练集，实体序列标注，训练BERT+BiLSTM+CRF
'''

data_type = "testing"
file = "nlpcc-iccpol-2016.kbqa."+data_type+"-data"
question_str = "<question"
triple_str = "<triple"
answer_str = "<answer"
start_str = "============="


q_t_a_list = []
seq_q_list = []    #["中","华","人","民"]
seq_tag_list = []  #[0,0,1,1]

with open(file) as f:
    q_str = ""
    t_str = ""
    a_str = ""

    for line in f:
        if question_str in line:
            q_str = line.strip()
        if triple_str in line:
            t_str = line.strip()
        if answer_str in line:
            a_str = line.strip()

        if start_str in line: #new question answer triple
            entities = t_str.split("|||")[0].split(">")[1].strip().decode("utf-8")
            q_str = q_str.split(">")[1].replace(" ","").strip().decode("utf-8")
            if entities in q_str:
                q_list = list(q_str)
                seq_q_list.extend(q_list)
                seq_q_list.extend([" "])
                tag_list = ["O" for i in range(len(q_list))]
                tag_start_index = q_str.find(entities)
                for i in range(tag_start_index, tag_start_index+len(entities)):
                    if tag_start_index == i:
                        tag_list[i] = "B-LOC"
                    else:
                        tag_list[i] = "I-LOC"
                seq_tag_list.extend(tag_list)
                seq_tag_list.extend([" "])
            else:
                pass
            q_t_a_list.append([q_str.encode("utf-8"), t_str, a_str])

print('\t'.join(seq_tag_list[0:50]))
print('\t'.join(seq_q_list[0:50]))
seq_result = [str(q.encode("utf-8"))+" "+tag for q, tag in zip(seq_q_list, seq_tag_list)]
with open("../NERdata/"+data_type+".txt", "w") as f:
    f.write("\n".join(seq_result))

df = pd.DataFrame(q_t_a_list, columns=["q_str", "t_str", "a_str"])
df.to_csv("q_t_a_df_"+data_type+".csv", index=False)