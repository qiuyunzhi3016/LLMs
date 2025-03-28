
""" 
实验室大模型（太一）的评估指标
"stiff neck; tightness in shoulders; muscle pain",
"no adverse reaction entities found.",
"changed my personality"
"""
import json
from collections import defaultdict
def str2dict(element):
    element = element.strip()
    if ';' in element:
        entities = []
        entities_raw_list = element.split('; ')
        for entity in entities_raw_list:
            entities.append(entity)
    else:
        entities = [element]
    golden_entities = {'AE': entities}
    return golden_entities


def com_prf(true_list, predict_list):                         # test_data_dir,pre_file_name,
    all_gold_label = defaultdict(set)
    all_pred_label = defaultdict(set)
    all_gold = []
    all_pred = []

    for idx, line in enumerate(true_list):
        label_dic = str2dict(line)                  # 真实样本       处理成这样{'AE': ['Leg pain', 'IVIG']}
        dic_str_pre = str2dict(predict_list[idx])   # 预测样本   处理成这样{'疾病': ['感染', 'IVIG']}

        for key in label_dic.keys():
            if key in all_gold_label.keys():
                pass
            else:
                all_gold_label[key] = []
            for value in label_dic[key]:
                all_gold_label[key].append(value+key+'idx'+str(idx))
                all_gold.append(value+key+'idx'+str(idx))
        try:
            pred_dic_item = dic_str_pre
            for pre_key in pred_dic_item.keys():
                if pre_key in all_pred_label.keys():
                    pass
                else:
                    all_pred_label[pre_key] = []
                for pre_value in pred_dic_item[pre_key]:
                    all_pred_label[pre_key].append(pre_value + pre_key+'idx'+str(idx))
                    all_pred.append(pre_value + pre_key+'idx'+str(idx))
        except:
            pass


    print('{0:^10}'.format("NER任务" +' 类别名称'),'\t','{0:^8}'.format('P'), '\t','{0:^8}'.format('R'), '\t','{0:^8}'.format('F1'))
    
    for type_name, true_entities in all_gold_label.items():
        pred_entities = all_pred_label[type_name]
        nb_correct = len(set(true_entities) & set(pred_entities))
        nb_pred = len(set(pred_entities))
        nb_true = len(set(true_entities))

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        print('{0:^10}'.format(type_name),'\t',format(p, '.4f'), '\t',format(r, '.4f'), '\t',format(f1, '.4f'))

    all_gold = []
    all_pred = []
    for i in all_gold_label:
        if all_pred_label[i] == set():
            all_pred_label[i] = []
        all_gold = all_gold+all_gold_label[i]
        all_pred = all_pred + all_pred_label[i]

    all_nb_correct = len(set(all_gold) & set(all_pred))
    all_nb_pred = len(set(all_pred))
    all_nb_true = len(set(all_gold))

    all_p = all_nb_correct / all_nb_pred if all_nb_pred > 0 else 0
    all_r = all_nb_correct / all_nb_true if all_nb_true > 0 else 0
    all_f1 = 2 * all_p * all_r / (all_p + all_r) if all_p + all_r > 0 else 0

    print('{0:^10}'.format('总类别'),'\t',format(all_p, '.4f'), '\t',format(all_r, '.4f'), '\t',format(all_f1, '.4f'))
    return all_p, all_r, all_f1

if __name__ =="__main__":

    predict_list = [
        #"stiff neck; tightness in shoulders; muscle pain",
        "no adverse reaction entities found.",
        #"changed my personality"
    ]

    true_list = [
        #"stiff neck",
        "no adverse reaction entities found.",
        #"no"
    ]

    com_prf(true_list, predict_list)
    print("######## END NER TEST ########\n")
    