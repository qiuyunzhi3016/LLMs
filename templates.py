
def tem(llm_name):
    a={}
    if llm_name=="llama2":
        a ={
            "system_format": '<<SYS>>\n{content}\n<</SYS>>\n\n',
            "user_format": '[INST]{content}[/INST]',
            "assistant_format": '{content} </s>',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '</s>'
        }
    elif llm_name=="llama3":
        a ={
            "system_format": '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
            "user_format": '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
            "assistant_format": '{content}<|eot_id|>',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '<|eot_id|>'
        }
    elif llm_name=="mistral":
        a ={
            "system_format": '<s>',
            "user_format": '[INST]{content}[/INST]',
            "assistant_format": '{content}</s>',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '</s>'
        }
    elif llm_name=="deepseek":
        a ={
            "system_format": "",
            "user_format": 'User: {content}\n\nAssistant: ',
            "assistant_format": '{content}<｜end▁of▁sentence｜>',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '<｜end▁of▁sentence｜>'
        }
    elif llm_name=="mixtral":
        a ={
            "system_format": '<s>',
            "user_format": '[INST]{content}[/INST]',
            "assistant_format": '{content}</s>',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '</s>'
        }
    elif llm_name=="qwen":
        a ={
            "system_format": '<|im_start|>system\n{content}<|im_end|>\n',
            "user_format": '<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
            "assistant_format": '{content}<|im_end|>\n',
            "system": ["Your task is to generate all adverse event entities in the follow text."],
            "stop_word": '<|im_end|>'
        }
    return a









