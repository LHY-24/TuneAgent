import argparse
import datasets
import json
import os

from verl.utils.hdfs_io import copy, makedirs


def download_file(url, local_path):
    print('download function is not implemented')
    exit()


def process_logs(source_dir):
    # Process config logs to training datas. Returns a dict
    question_list = []
    answer_list = []
    type_list = []
    task_list = []

    for files in os.listdir(source_dir):
        if not files.endswith(".log"):
            continue
        with open(source_dir + "/" + files, "r") as f:
            task = f.readline()
            qa_instances = f.readlines()
        for qa_json_instance in qa_instances:
            qa_instance = json.loads(qa_json_instance)
            qa_type, question = qa_instance["question"].split("\t")
            # if qa_type != "Bool":
            #     continue
            answer = qa_instance["answer"]
            question_list.append(question)
            answer_list.append(qa_type + "\t" + json.dumps(answer))
            type_list.append(qa_type)
            task_list.append(task)

    return {
        'question': question_list,
        'answer': answer_list,
        'type': type_list,
        'task': task_list
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/tuneagent')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=12800,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of validation samples to use')
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # train_url = ""
    # validate_url = ""

    train_dir = os.path.join(local_dir, "tuneagent_train")
    validate_dir = os.path.join(local_dir, "tuneagent_validate")

    # train_file = os.path.join(local_dir, "tuneagent_train.json")
    # validate_file = os.path.join(local_dir, "tuneagent_validate.json")

    # if not os.path.exists(train_file):
    #     print(f"Downloading training data to {train_file}...")
    #     download_file(train_url, train_file)

    # if not os.path.exists(validate_file):
    #     print(f"Downloading training data to {validate_file}...")
    #     download_file(validate_url, validate_file)

    # print("Loading downloaded files...")
    # with open(train_file, 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    
    # with open(validate_file, 'r', encoding='utf-8') as f:
    #     validation_data = json.load(f)

    train_dataset = datasets.Dataset.from_dict(process_logs(train_dir))
    validation_dataset = datasets.Dataset.from_dict(process_logs(validate_dir))

    # train_dataset = datasets.Dataset.from_dict({
    #     'question': [item['query'] for item in train_data],
    #     'answer': [item['result'] for item in train_data],
    # })
    # validation_dataset = datasets.Dataset.from_dict({
    #     'question': [item['query'] for item in validation_data],
    #     'answer': [item['result'] for item in validation_data],
    # })

    # answer okay
    qa_bool_instruction = """Given TARGET = {}. I want to explore the config options related to TARGET in the Linux kernel configurations. I will give you some configs, and you should determine whether they will increase or decrease TARGET, or not related to TARGET:
    (1) If a config increases TARGET, out[CONFIG_NAME increase]. 
    (2) If it decreases TARGET, output [CONFIG_NAME decrease]. 
    (3) If it is not related to TARGET, output [CONFIG_NAME - cannot determine impact without specific context]. 
    Each single answer should be given wihout any explanation in pure text form. All the answers should be wrapped inside <answer>...</answer>. For example, if you are given configs \"64-bit kernel (64BIT)\nMitigations for speculative execution vulnerabilities (SPECULATION_MITIGATIONS)\nVirtualization (VIRTUALIZATION)\nEnable loadable module support (MODULES)\nEnable the block layer (BLOCK)\nNetworking support (NET)\", you should answer \"<answer>[64BIT increase]\n[SPECULATION_MITIGATIONS decrease]\n[VIRTUALIZATION decrease]\n[MODULES - cannot determine impact without specific context]\n[BLOCK - cannot determine impact without specific context]\n[NET - cannot determine impact without specific context]</answer>\". 
    Also, you can use the tools provided to you to answer the question, and tool call should be wrapped inside <tool_call>...</tool_call>. You can use the tool as many times as you want.
    You should first conduct reasoning first, and it should be wrapped inside <think>...</think>. Then you can use use the tool if needed. Notice, <think>...</think><tool_call></tool_call> pair can be used multiple times, which means you can call the tools as many times as you need. Then you need to reasoning again, then output the answer inside <answer>...</answer>.
    You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>. When you have the final answer, you should output the answer inside <answer>...</answer>. More important, <think>...</think><answer>...</answer> pair can only appear once.
    Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""

# For example, a well formatted output should be like this:
#     <think>
#     ...
#     </think>
#     <tool_call>
#     ...
#     </tool_call>
#     (repeat several times or none)
#     <think>
#     ...
#     </think>
#     <answer>
#     ...
#     </answer>

    instruction_header_bool = """Given TARGET = {}. You need to explore the config options related to TARGET in the Linux kernel configs. I will give you some configs, and you should determine whether they will increase or decrease TARGET, or not related to TARGET.
"""
    instruction_header_menu = """Given TARGET = {}. You need to explore the config directories related to TARGET in the Linux kernel configs. I will give you some config directories, and you should determine whether they will affect TARGET.
"""
    instruction_header_choice = """Given TARGET = {}. You need to explore the config options related to TARGET in the Linux kernel configs. I will give you some choices of a config, and you need to choose which config is most likely related to TARGET.
"""
    instruction_header_value = """Given TARGET = {}. You need to explore the config options related to TARGET in the Linux kernel configs. I will list some numeric config options along with their corresponding value ranges. For each option, you need to select a value that may help improve TARGET. If the option is not related to TARGET, reset it to the default value.
"""
    instruction_header = {
        "Bool": instruction_header_bool,
        "Menu": instruction_header_menu,
        "Choice": instruction_header_choice,
        "Value": instruction_header_value
    }

    instruction_mid = """You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. You must use the tools to gather more information about the configs, and you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.
Notice! The answer inside <answer>...</answer> must follow these rules:
"""

    instruction_tail_bool = """(1) If a config increases TARGET, output [CONFIG_NAME - increase]. 
(2) If it decreases TARGET, output [CONFIG_NAME - decrease]. 
(3) If it is not related to TARGET, output [CONFIG_NAME - cannot determine impact without specific context].
(4) Each single answer should be given wihout any explanation in pure text form.
"""
    instruction_tail_menu = """(1) If a directory may concern with TARGET, output [DIRECTORY_NAME]
(2) Each single answer should be given wihout any explanation in pure text form.
"""
    instruction_tail_choice = """(1) The config you chose should output [CONFIG_NAME]
(2) Only one config can be given, and it should be given without any explanation in pure text form.
"""
    instruction_tail_value = """(1) For each config given to you, output CONFIG_NAME (recommended value)
(2) Each single answer should be given without any explanation in pure text form.
"""
    instruction_tail = {
        "Bool": instruction_tail_bool,
        "Menu": instruction_tail_menu,
        "Choice": instruction_tail_choice,
        "Value": instruction_tail_value
    }

    instruction_example_bool = """For example, if you are given configs \"64-bit kernel (64BIT)\nMitigations for speculative execution vulnerabilities (SPECULATION_MITIGATIONS)\nVirtualization (VIRTUALIZATION)\nEnable loadable module support (MODULES)\nEnable the block layer (BLOCK)\nNetworking support (NET)\", you can answer \"<answer>[64BIT increase]\n[SPECULATION_MITIGATIONS decrease]\n[VIRTUALIZATION decrease]\n[MODULES - cannot determine impact without specific context]\n[BLOCK - cannot determine impact without specific context]\n[NET - cannot determine impact without specific context]</answer>\". 
"""
    instruction_example_menu = """For example, if you are given config menus \"0 Magic SysRq key (MAGIC_SYSRQ)\n1 Debug Filesystem (DEBUG_FS)\n2 KGDB: kernel debugger (KGDB)\", you can answer \"<answer>[Debug Filesystem]\n[KGDB: kernel debugger]</answer>\"
"""
    instruction_example_choice = """For example, if you are given configs \"port 0x80 based port-IO delay [recommended] (IO_DELAY_0X80)\nport 0xed based port-IO delay (IO_DELAY_0XED)\nudelay based port-IO delay (IO_DELAY_UDELAY)\nno port-IO delay (IO_DELAY_NONE)\", you can answer \"<answer>[IO_DELAY_NONE]</answer>\".
"""
    instruction_example_value = """For example, if you are given configs \" Default console loglevel (1-15) (7)\nquiet console loglevel (1-15) (4)\nDefault message log level (1-7) (4)\", you can answer\"<answer>Default console loglevel (1-15) (7)\nquiet console loglevel (1-15) (4)\nDefault message log level (1-7) (4)</answer>\".
"""
    instruction_example = {
        "Bool": instruction_example_bool,
        "Menu": instruction_example_menu,
        "Choice": instruction_example_choice,
        "Value": instruction_example_value
    }

    instruction_format = """
Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>"""

    instruction_following = """Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""

    qa_bool_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. Please choose the configs concerned with TARGET in the CONFIGS as much as possible. For each concerned config related to TARGET, you should determine whether it will increase or decrease TARGET. If it increases TARGET, output [CONFIG increase]. If it decreases TARGET, output [CONFIG decrease]. If a config is not related to TARGET, output [CONFIG - cannot determine impact without specific context]. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, give me the config names in my given CONFIGS, each line represents a single config like this:\n[config_name_1 increase]\n[config_name_2 decrease]\n....\n[config_name_n increase]\nBelow are the numeric config options for your recommendations: \n"
    qa_menu_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. Please choose the directories concerned with TARGET in the DIRECTORIES as much as possible. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, give me directory names with index in my given DIRECTORIES, each line represents a single directory like this:\n[directory_name_1]\n[directory_name_2]\n....\n[directory_name_n]\nBelow are the numeric config options for your recommendations: \n"
    qa_choice_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. The CONFIGS I gave you are choices of a config, and you need to choose which config is most likely related to TARGET. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, each line represents a single config like this:\n[config_name]\nBelow are the numeric config options for your recommendations: \n"
    qa_value_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. I have listed some numeric config options listed in menuconfig, along with their corresponding value ranges. For each option, please select a value that may help improve TARGET. If the option is not related to TARGET, reset it to the defalut value. Config input format: [option name] (default value). Value output format: [option name] (recommended  value). For instance, if you are given: 'maximum CPU number(1=>2 2=>4)  (cpunum) (1), Your response would be: 'maximum CPU number(1=>2 2=>4)  (cpunum) (2). Because when the CPU number is more, the speed is usually better. Below are the numeric config options for your recommendations:\n"
    qa_type_prompt = {
        "Bool": qa_bool_prompt,
        "Menu": qa_menu_prompt,
        "Choice": qa_choice_prompt,
        "Value": qa_value_prompt,
    }

    def make_map_fn():
        def process_fn(content):
            task = content.pop('task')
            qa_type = content.pop('type')
            # question_raw = content.pop('question')
            # question = instruction_following + \
            #            "Question: Given TARGET = " + \
            #            task + \
            #            qa_type_prompt[qa_type] + \
            #            question_raw
            # if qa_type == "Bool":
            #     question = instruction_header_bool.format(task) + instruction_mid + instruction_tail_bool + instruction_example_bool + instruction_format + "Here are the given configs:\n" + content.pop("question")
            #     # question = qa_bool_instruction.format(task) + "Here are the given configs:\n" + content.pop("question")
            # else:
            #     question_raw = content.pop('question')
            #     question = instruction_following + \
            #            "Question: Given TARGET = " + \
            #            task + \
            #            qa_type_prompt[qa_type] + \
            #            question_raw
            question = instruction_header[qa_type].format(task) + instruction_mid + instruction_tail[qa_type] + instruction_example[qa_type] + instruction_format + "Here are the given configs:\n" + content.pop("question")

            answer = content.pop('answer')
            
            data = {
                "data_source": 'tuneagent',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "linux_config_autogen",
                "reward_model": {
                    "ground_truth": answer
                },
                "extra_info": {
                    "type": qa_type
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn())
    validation_dataset = validation_dataset.map(function=make_map_fn())

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)