from datasets import Dataset
import pandas as pd
from transformers import (HfArgumentParser, AutoTokenizer,AutoModelForCausalLM,DataCollatorForSeq2Seq,TrainingArguments,Trainer, )
import torch
torch.backends.cuda.enable_flash_sdp(False)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from evaluator import com_prf
from tqdm import tqdm
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import random
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 忽略告警
 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Template:
    system_format = '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    assistant_format = '{content}<|eot_id|>'
    system = [

        "Your task is to generate all adverse reaction entities in the follow text.",
        "Kindly assist in extracting all adverse drug reaction entities from the sentences.",
        "Please help in identifying all adverse drug reaction entities mentioned in the sentences.",
        "I would appreciate it if you could extract all adverse drug reaction entities from the sentences.",
        "Could you please extract all adverse drug reaction entities from the sentences?",
        "I need assistance in extracting all adverse drug reaction entities from the sentences.",
        "Can you help me identify all adverse drug reaction entities from the sentences?",
        "Seeking help to extract all adverse drug reaction entities from the sentences.",
        "I am looking to identify all adverse drug reaction entities present in the sentences.",
        "Assistance is required in extracting all adverse drug reaction entities from the sentences.",
        "Extract all adverse drug reaction entities from the sentences, please.",     
        "I am seeking assistance to identify all adverse drug reaction entities in the sentences.",
        "Please help me identify all adverse drug reaction entities mentioned in the sentences.",
        "I am looking for support in identifying all adverse drug reaction entities from the sentences.",
        "I require assistance in extracting all adverse drug reaction entities from the sentences.",
        "Please aid in extracting all adverse drug reaction entities from the sentences.",  
    ]
    stop_word = '<|eot_id|>'


""" 
找到所有的全连接层
"""
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    return lora_module_names


"""
一些自定义参数
"""
@dataclass
class DataTrainingArguments:
    llm_path: Optional[str] = field(default=None, metadata={"help": "llm的路径"})
    train_path: Optional[str] = field(default=None, metadata={"help": "训练集路径"})
    dev_path: Optional[str] = field(default=None, metadata={"help": "验证集路径"})
    test_path: Optional[str] = field(default=None, metadata={"help": "测试集路径"})
    lora_path: Optional[str] = field(default=None, metadata={"help": "lora模型路径"})
    merge_path: Optional[str] = field(default=None, metadata={"help": "merge模型路径"})
    max_seq_length: Optional[int] = field(default=512,metadata={"help": "句子最大长度"})
    
@dataclass
class ModelArguments:
    # lora部分参数
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


# 初始化数据和模型参数
parser = argparse.ArgumentParser()
parser.add_argument("--train_args_file", type=str, default='/root/autodl-tmp/llm_lora/llama3-8b-sft-lora.json', help="")
args = parser.parse_args()
train_args_file = args.train_args_file
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file=train_args_file)    

# 加载和处理数据
train_dataset = Dataset.from_pandas(pd.read_json(data_args.train_path, lines=True))
eval_dataset = Dataset.from_pandas(pd.read_json(data_args.dev_path, lines=True))
tokenizer = AutoTokenizer.from_pretrained(data_args.llm_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(data_args.llm_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16,)
base_model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

def collate_fn(examples):
    prompts = []
    input_part_targets_len = []
    for example in examples:
        # 获得知识
        if len(example["knowledge"]) == 0:
            knowledge = "no external knowledge"
        else:
            know = []
            for row in example["knowledge"]:
                if row["definition"] == "no":
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"])
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ", the corresponding semantic types is " + row["semantic type"])
                    know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"])
                else:
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ', the corresponding definition is ' + row["definition"])
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ", the corresponding semantic types is " + row["semantic type"] + ', the corresponding definition is ' + row["definition"])
                    know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ', the corresponding definition is ' + row["definition"])
            knowledge_str = ';'.join(know)
            knowledge = knowledge_str


        system_text = Template.system_format.format(content=random.choice(Template.system) + "Please refer to the following external knowledge:\n ###" + knowledge)  # instruction
        human_text = Template.user_format.format(content=example["context"])   # input
        if len(example["response"]) == 0:
            resposrs = "no adverse reaction entities found."                   # output
        else:
            resposrs = example["response"][0]["ADR"]
        assistant_text = Template.assistant_format.format(content=resposrs)
        prompts.append(system_text + human_text + assistant_text)
        input_part_targets_len.append(len(tokenizer.tokenize(system_text + human_text)) + 1)  # +1 is bos token
    llm_inputs = tokenizer(prompts,max_length=data_args.max_seq_length + 100,padding="longest",return_tensors="pt",truncation=True,)
    targets = llm_inputs.input_ids.masked_fill(llm_inputs.input_ids == tokenizer.pad_token_id, -100)
    for i, l in enumerate(input_part_targets_len):
        targets[i][:l] = -100
    return {
        "input_ids": llm_inputs.input_ids,
        "attention_mask": llm_inputs.attention_mask,
        "labels": targets,
        }

# 开始训练
target_modules = find_all_linear_names(base_model)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules = target_modules,
    inference_mode=False,  # 训练模式
    r=model_args.lora_rank,  # Lora 秩
    lora_alpha=model_args.lora_alpha,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=model_args.lora_dropout  # Dropout 比例
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()

# 计算所有参数和可训练参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

# 开始训练并保存lora模型 以及merge模型
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model(data_args.lora_path)    # 保存模型和训练器的状态，包括训练过程相关信息
    # trainer.model.save_pretrained()          仅保存模型的权重和配置文件


    """ checkpoint = [file for file in os.listdir(training_args.output_dir) if 'checkpoint' in file][-1] #选择更新日期最新的检查点
    checkpoint_path = training_args.output_dir + "/" + checkpoint
    trainer.save_model(checkpoint_path)  # Saves the tokenizer too """
    # 保存训练的评估指标
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()   

    # 执行合并操作,# 保存微调模型
    merge_model = PeftModel.from_pretrained(
        base_model, 
        data_args.lora_path, 
        torch_dtype=torch.float16,
        config=config,
        )
    model1 = merge_model.merge_and_unload()

    tokenizer.save_pretrained(data_args.merge_path)
    model1.save_pretrained(data_args.merge_path)

# 开始验证过程
if training_args.do_eval:
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


# 预测过程
if training_args.do_predict: 
    # 清除缓存       
    del trainer
    del base_model
    del model
    if training_args.do_train:
        del model1
        del merge_model
        
    merged_model = AutoModelForCausalLM.from_pretrained(    
        data_args.merge_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).eval()
    merged_model = merged_model.to("cuda")
    merged_model.config.use_cache = True  # use for inference

    # 加载处理测试集
    test_dataset = Dataset.from_pandas(pd.read_json(data_args.test_path, lines=True))
    grount_labels = [] 
    for row in test_dataset['response']:
        if len(row) == 0:
            grount_labels.append("no adverse reaction entities found.")
        else:
            grount_labels.append(row[0]["ADR"])

    def evaluate(input_ids,attention_mask):
        with torch.no_grad():
            outputs = merged_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=128, 
                do_sample=False,
                #top_p=top_p, 
                temperature=0.0, 
                repetition_penalty=1.0,

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        return response




    predict_list = []
    for example in tqdm(test_dataset, desc="Generating predictions", total=len(test_dataset)):
        # 获得知识
        if len(example["knowledge"]) == 0:
            knowledge = "no external knowledge"
        else:
            know = []
            for row in example["knowledge"]:
                if row["definition"] == "no":
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"])
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ", the corresponding semantic types is " + row["semantic type"])
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"])
                    #pass
                    know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"])
                else:
                    know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ', the corresponding definition is ' + row["definition"])
                    #know.append("For the token span " + row["token span"] + ", the corresponding UMLS CUI is " + row["UMLS CUI"] + ", the corresponding semantic types is " + row["semantic type"] + ', the corresponding definition is ' + row["definition"])
                    #know.append("For the token span " + row["token span"] + ', the corresponding definition is ' + row["definition"])
            knowledge_str = ';'.join(know)
            knowledge = knowledge_str


        # 处理输入
        system_text = Template.system_format.format(content=random.choice(Template.system) + "Please refer to the following external knowledge:\n ###" + knowledge)  # instruction
        human_text = Template.user_format.format(content=example["context"])   # input
        #system_text = Template.system_format.format(content=random.choice(Template.system))
        #human_text = Template.user_format.format(content=example["context"] + "Please refer to the following external knowledge:\n ###" + knowledge)   # input
        #human_text = Template.user_format.format(content=example["context"])
        instruction1 = system_text + human_text
        instruction = tokenizer(instruction1, return_tensors="pt", add_special_tokens=False).to("cuda")   # 张量形式
        response = evaluate(instruction.input_ids, instruction.attention_mask)
        predict_list.append(response)

    results = com_prf(predict_list, grount_labels)
    print("Results:")
    print(results)
    with open(os.path.join(training_args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
    with open(os.path.join(training_args.output_dir, "predict_list.txt"), 'w') as f:
        f.write('\n'.join(predict_list))


