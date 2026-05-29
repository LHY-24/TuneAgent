import re
import os
import traceback

from openai import OpenAI

def remove_last_parentheses(text):
    return re.sub(r'\([^()]*\)(?=[^()]*$)', '', text)

def extract_answer(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

def extract_target(solution_str):
    """Extract target from the solution string."""
    target_pattern = r'Given TARGET = (.*?)\. You need'
    match = re.search(target_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_origin_config(solution_str):
    """Extract the given config list from prompt"""
    answer_pattern = r'Here are the given configs:(.*?)<\|im_end\|>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def ask(client, content):
    conversation = [
        {
            "role": "user",
            "content": content
        }
    ]
    response = client.chat.completions.create(
        messages=conversation, model="gpt-4o-mini"
    )
    return response.choices[0].message.content

menu_prompt = """I'm optimizing linux kernel, and my target is '{}'. I want to reach this target by adjusting linux kernel config, and I want to ask you if config '{}' may affects my target? Answer my question using only 'yes' or 'no'. Do not explain."""

def ask_menu(client, target, origin_config, answer):
    if answer.strip() == "":
        return 0.0
    """
    iterate each menu item in origin menus, ask gpt if they are related to target.
    then check if answer gives the same result
    """
    origin_configs = origin_config.split('\n')
    answers = answer.split('\n')

    def remove_bracket(s):
        s = s.strip().lower()
        if s[0] == '[' and s[-1] == ']':
            return s[1:-1].strip()
        return s
    
    origin_configs = list(map(remove_bracket, origin_configs))
    answers = list(map(remove_bracket, answers))
    
    correct_num = 0
    error_num = 0
    for origin_config in origin_configs:
        result:str = ask(client, menu_prompt.format(target, origin_config)).strip().lower()
        # parse result
        if result.endswith('.'):
            result = result[:-1]
        if result != 'yes' and result != 'no':
            correct_num += 1
            continue
        result_bool = False
        if result == 'yes':
            result_bool = True
        # check if this config exists in answer
        in_answer = False
        if origin_config in answers:
            in_answer = True
        else:
            # try to remove barcket of origin_config
            if remove_last_parentheses(origin_config) in answers:
                in_answer = True
        if in_answer ^ result_bool:
            error_num += 1
        else:
            correct_num += 1
    score = 1.0 * correct_num / (correct_num + error_num)
    return score

bool_prompt = """I'm optimizing linux kernel, and my target is '{}'. I want to reach this target by adjusting linux kernel config, and I want to ask you if config '{}' affects my target? Answer my question using 'yes' if it may increases target, or 'no' if it may decreases target, or 'unknown' if it does not related to target. Do not explain."""

def ask_bool(client, target, origin_config, answer):
    if answer.strip() == "":
        return 0.0
    """
    iterate each config and ask llm about their value
    """
    origin_configs = origin_config.split('\n')
    answers = answer.split('\n')
    answer_dict = {}

    def remove_bracket(s):
        s = s.strip().lower()
        if s[0] == '[' and s[-1] == ']':
            return s[1:-1].strip()
        return s
    
    def parse_answer(s):
        s = s.strip().lower()
        if s[0] == '[' and s[-1] == ']':
            s = s[1:-1]
        l = s.split('-')
        if len(l) < 2:
            return
        config = l[0].strip()
        result = l[1].strip()
        result_int = 0
        if result == 'increase':
            result_int = 0
        elif result == 'decrease':
            result_int = 1
        else:
            result_int = 2
        answer_dict[config] = result_int
    
    origin_configs = list(map(remove_bracket, origin_configs))
    for answer in answers:
        parse_answer(answer)

    correct_num = 0
    error_num = 0
    for origin_config in origin_configs:
        if origin_config not in answer_dict.keys():
            error_num += 1
            continue
        result:str = ask(client, bool_prompt.format(target, origin_config)).lower().strip()
        # parse result
        if result.endswith('.'):
            result = result[:-1]
        if result != 'yes' and result != 'no' and result != 'unknown':
            correct_num += 1
            continue
        result_int = 0
        if result == 'yes':
            result_int = 0
        elif result == 'no':
            result_int = 1
        else:
            result_int = 2
        
        if result_int != answer_dict[origin_config]:
            error_num += 1
        else:
            correct_num += 1
    score = 1.0 * correct_num / (correct_num + error_num)
    return score

choice_prompt = """I'm optimizing linux kernel, and my target is '{}'. I want to reach this target by adjusting linux kernel config, and among these config options '{}', which one may help improve the target? Give me the config name without any explain."""

def ask_choice(client, target, origin_config, answer):
    if answer.strip() == "":
        return 0.0
    origin_configs = origin_config.split('\n')
    
    def remove_bracket(s):
        s = s.strip().lower()
        if s[0] == '[' and s[-1] == ']':
            return s[1:-1].strip()
        return s
    origin_configs = list(map(remove_bracket, origin_configs))
    answer = remove_bracket(answer)

    configs_name_str = ','.join(origin_configs)
    result:str = ask(client, choice_prompt.format(target, configs_name_str)).lower().strip()
    if result.startswith('none'):
        return 1.0
    if result == answer:
        return 1.0
    return 0.0

value_prompt = """I'm optimizing linux kernel, and my target is '{}'. I want to reach this target by adjusting linux kernel config, and I want to ask you if config '{}' affects my target? If this config does not affect my target, answer "default", otherwise give me the value of the config. Do not explain."""

def ask_value(client, target, origin_config, answer):
    if answer.strip() == "":
        return 0.0
    origin_configs = origin_config.split('\n')
    answers = answer.split('\n')

    def extract(s):
        match = re.search(r'^(.*?)\((.*?)\)', s)
        if match is None:
            return
        return(match.group(1).strip(), match.group(2).strip())
    
    # process origin_configs
    origin_configs_dict = {}
    for origin_config in origin_configs:
        res = extract(origin_config)
        if res is None:
            continue
        origin_configs_dict[res[0]] = res[1]
    # process answer
    answers_dict = {}
    for answer in answers:
        res = extract(answer)
        if res is None:
            continue
        answers_dict[res[0]] = res[1]
    
    correct_num = 0
    error_num = 0
    # print("[DEBUG] origin config: ", origin_config, "; answer: ", answer, "; origin dict: ", origin_configs_dict, "; answer dict: ", answers_dict)
    for origin_config in origin_configs_dict.keys():
        if origin_config not in answers_dict.keys():
            error_num += 1
            continue
        result:str = ask(client, value_prompt.format(target, origin_config)).lower().strip()
        if result.startswith('default'):
            if answers_dict[origin_config] == origin_configs_dict[origin_config]:
                correct_num += 1
            else:
                error_num += 1
        else:
            if answers_dict[origin_config] == result:
                correct_num += 1
            else:
                error_num += 1
    return 1.0 * correct_num / (correct_num + error_num)
        

def compute_score_answer(solution_str, ground_truth):
    if solution_str is None:
        return .0
    
    origin_config = extract_origin_config(solution_str)
    target = extract_target(solution_str)
    answer = extract_answer(solution_str)

    # initialize openai
    client = OpenAI(base_url=os.environ['OPENAI_BASE_URL'], api_key=os.environ['OPENAI_API_KEY'])

    process_ty_dict = {
        "Bool": ask_bool,
        "Menu": ask_menu,
        "Choice": ask_choice,
        "Value": ask_value
    }
    qa_type, answer_json = ground_truth.split("\t")

    try:
        score = process_ty_dict[qa_type](client, target, origin_config, answer)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    print(score)
    return score

def compute_score_format(solution_str):
    if solution_str is None:
        return .0
    
    try:
        # Perfect format match for the new structure
        # First <|im_start|>assistant should have <think> and possibly <tool_call>
        # Then <|im_start|>tool with <tool_response> (can repeat with assistant/tool pairs)
        # Final <|im_start|>assistant with the answer and <|im_end|>
        
        # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

        format_reward = 0.0
        
        # If no blocks found, return 0
        if not assistant_blocks:
            return 0.0
        
        assistant_block_has_score = False
        
        # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
        # Check first assistant block contains <think> tags
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>\n<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
                # soft_think_match = re.search(r'<think>(.*?)</think>(.*?)<tool_call>(.*?)</tool_call>', assistant_block, re.DOTALL)
                if think_match and not assistant_block_has_score:
                    # format_reward += 0.2 * (0.8 ** i)
                    assistant_block_has_score = True
                    format_reward += 0.45

        # Check the last assistant block contains <answer> tags
        if assistant_blocks:  # 确保有至少一个assistant块
            last_assistant_block = assistant_blocks[-1]
            think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.55
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format: {e}")
        return 0.0
    
    return format_reward

def compute_score_format_answer(solution_str, ground_truth):
    if solution_str is None or ground_truth is None:
        return 0.0

    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)

        format_reward = min(format_reward, 1.0)
        reward = .0
        if format_reward >= 0.99:
            reward = -1.0 + format_reward + answer_reward
        else:
            reward = -1.0 + format_reward
        # with open("outputs/solution.log", "a+") as f:
        #     f.write("SOLUTION\n" + solution_str + "\nTRUTH\n" + ground_truth +"\nREWARD\n" + str(reward) + "\n\n\n")
        return reward
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return 0.0
