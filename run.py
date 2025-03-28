""" 
能运行，调整了一下格式,第二个输出不行

"""

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
from templates import tem
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # 忽略告警


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

""" 
找到所有的全连接层
"""
def find_all_linear_names(model):
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
    data_name: Optional[str] = field(default=None, metadata={"help": "数据集名字"})
    train_path: Optional[str] = field(default=None, metadata={"help": "训练集路径"})
    dev_path: Optional[str] = field(default=None, metadata={"help": "验证集路径"})
    test_path: Optional[str] = field(default=None, metadata={"help": "测试集路径"})
    llm_path: Optional[str] = field(default=None, metadata={"help": "llm的路径"})
    prompt_format: Optional[str] = field(default=None, metadata={"help": "提示词格式选择"})
    lora_path: Optional[str] = field(default=None, metadata={"help": "lora模型路径"})
    merge_path: Optional[str] = field(default=None, metadata={"help": "merge模型路径"})
    max_seq_length: Optional[int] = field(default=512,metadata={"help": "句子最大长度"})
@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "模型名字"},)
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})


# 初始化数据和模型参数
parser = argparse.ArgumentParser()
parser.add_argument("--train_args_file", type=str, default='/root/autodl-tmp/llm_lora/llm-sft-lora.json', help="")
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


# 模板
Template_llm = tem(data_args.prompt_format)
""" system_format = Template_llm['system_format'].format(content="ssss")    # 系统背景
user_format = Template_llm['user_format'].format(content="ssss")     # 提示词+输入
assistant_format = Template_llm['assistant_format'].format(content="ssss")   # 标签格式 """

def process_func(example):
    MAX_LENGTH = data_args.max_seq_length    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    system_format = Template_llm['system_format'].format(content="ner task")  # 系统背景
    user_format = Template_llm['user_format'].format(content=random.choice(Template_llm['system']) + example['context'])     # 提示词+输入
    instruction = tokenizer(system_format+user_format, add_special_tokens=False)

    #instruction = tokenizer(f"User: {random.choice(Template.system) + example['context']}\n\nAssistant:\n", add_special_tokens=False)
    if len(example["response"]) == 0:
        resposrs = "no adverse event entities found."
    else:
        resposrs = example["response"][0]["ADR"]
    response = tokenizer(Template_llm['assistant_format'].format(content=resposrs), add_special_tokens=False)
    #response = tokenizer(f"{resposrs}<｜end▁of▁sentence｜>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)

# 开始训练
target_modules = find_all_linear_names(base_model)
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #target_modules = target_modules,
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
    #data_collator=collate_fn,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练并保存lora模型 以及merge模型
if training_args.do_train:
    train_result = trainer.train()
    trainer.save_model(data_args.lora_path)    # 保存模型和训练器的状态，包括训练过程相关信息
    # trainer.model.save_pretrained()          仅保存模型的权重和配置文件

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
            grount_labels.append("no adverse event entities found.")
        else:
            grount_labels.append(row[0]["ADR"])

    def evaluate(input_ids,attention_mask):
        with torch.no_grad():
            outputs = merged_model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=20, 
                do_sample=False,   # 设置是否使用采样方法生成文本。如果为True，则使用采样方法，否则使用贪婪解码方法。
                #top_p=0.8,        # 设置采样时，保留概率累积的阈值。在采样时，会选择率累积超过这个值的最高概率的词.
                temperature=0.0,     # 设置生成文本时的多样性。较高的温度值会使生成的文本更加多样化，但可能会牺牲一些语义连贯性。
                #repetition_penalty=1.8,    # 设置生成文本中重复内容的惩罚因子。较高的值会降低生成文本中重复内容的可能性。

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        response = response.strip().replace(tokenizer.eos_token, "").strip()
        return response

    predict_list = []
    final_dict = []
    for example in tqdm(test_dataset, desc="Generating predictions", total=len(test_dataset)):
        MAX_LENGTH = data_args.max_seq_length    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask = [], []
        system_format = Template_llm['system_format'].format(content="ner task")  # 系统背景
        user_format = Template_llm['user_format'].format(content=random.choice(Template_llm['system']) + example['context'])     # 提示词+输入
        instruction = tokenizer(system_format+user_format, return_tensors="pt", add_special_tokens=False).to("cuda")
        input_ids = instruction["input_ids"] 
        attention_mask = instruction["attention_mask"]   # 因为eos token咱们也是要关注的所以 补充为1
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        response = evaluate(input_ids, attention_mask)
        context_yuan = example['context']
        true_label = example['response']
        if len(true_label) == 0:
            true_label = "no adverse reaction entities found."
        else:
            true_label = true_label[0]["ADR"]
        
        final_dict.append({"true_entities":true_label, "predicted_entities":response, "context":context_yuan, })


        predict_list.append(response)

    results = com_prf(predict_list, grount_labels)
    print(results)
    with open("/root/autodl-tmp/llm_lora/results/"+data_args.data_name+"/"+model_args.model_name+".jsonl", 'w', encoding='utf-8') as file:
        for dii in final_dict:
            json_line = json.dumps(dii, ensure_ascii=False)  # 转换为 JSON 字符串
            file.write(json_line + '\n')  # 写入文件，每个 JSON 对象占一行


