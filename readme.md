HHHHH

本项目用于使用lora技术进行微调llm
LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：
{
  "instruction": "回答以下用户问题，仅输出答案。",
  "input": "1+1等于几?",
  "output": "2"
}
其中，instruction 是用户指令，告知模型其需要完成的任务；input 是用户输入，是完成用户指令所必须的输入内容；output 是模型应该给出的输出。

run.py  
run_knowledge.py   两者的区别在于一个引入知识，一个没有，但是目前没有跑这个run_knowledge.py这个文件

# 一：文件说明

--args
  llm-sft-lora.json            # 模型参数设置文件
--output                        # 输出路径，可以自动生成这个路径
  checkpoint-{num}
--result                        # 专门存储每个模型的预测结果，包括真实标签，原始句子
  cadec
  smm4h2020
  vaccine
--evaluator.py                  # 评估函数
--templates.py                  # 各个模型的提示模板不同，需要更换
--readme.md                     # 项目说明文件
--requirements.txt              # 项目相关包版本文件
--run.py                        # 主函数文件
--run_knowledge.py              # 两者的区别在于一个引入知识，但是目前没有跑这个文件