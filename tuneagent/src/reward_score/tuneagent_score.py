import re
import json
import pdb
import traceback
from pathlib import Path

# def content_check(solution, ground_truth):
#     def extract_configs(config_str):
#         configs = config_str.split('\n')
#         config_set = set()
#         config_value_dict = {}
#         for config in configs:
#             config_name, value = config.split('=')
#             if value is None:
#                 continue
#             config_set.add(config_name)
#             config_value_dict[config_name] = value
#         return config_set, config_value_dict
    
#     solution_cs, solution_cvd = extract_configs(solution)
#     truth_cs, truth_cvd = extract_configs(ground_truth)
    
#     union_set = solution_cs & truth_cs
#     # score: 0.4 for name matching, 0.6 for value matching
#     score = len(union_set) / len(truth_cs)
#     correct_value = 0
#     for config in union_set:
#         if solution_cvd[config] == truth_cvd[config]:
#             correct_value += 1
#     score *= correct_value / len(union_set)
#     return score

def content_check(solution, ground_truth):
    
    def process_bool(answer, truth_json):
        score = 0.0
        # extract truth
        truth = json.loads(truth_json)
        truth_dict = {}
        for t in truth:
            truth_dict[t["config"]] = t["value"]
        value2str = {
            0: "decrease",
            2: "increase",
            -1: "cannot determine impact without specific context"
        }
        correct_num = 0
        error_num = 0
        # extract solution
        answers = answer.split("\n")
        configs = []
        # define format score
        bracket_correct = 0 # score 0.1
        split_correct = 0 # score 0.1
        value_correct = 0 # score 0.1
        for ans in answers:
            if len(ans) == 0:
                continue
            # check bracket
            if ans[0] != '[' or ans[-1] != ']':
                bracket_correct = -1
            else:
                ans = ans[1:-1]
            # check split
            split_res = ans.split('-')
            if len(split_res) != 2:
                split_correct = -1
                continue
            # check value
            config, res = split_res
            config = config.strip()
            res = res.strip()
            if res not in ["increase", "decrease", "cannot determine impact without specific context"]:
                value_correct = -1
            else:
                if value_correct == 0:
                    value_correct = 1
                configs.append([config, res])
        # check answer && scoring
        for i in range(len(configs)):
            config_name, config_value = configs[i]
            config_name = config_name.lower()
            if config_name in truth_dict.keys():
                if value2str[truth_dict[config_name]] == config_value:
                    correct_num += 1
                del truth_dict[config_name]
            else:
                error_num += 1
        
        # add to score
        if bracket_correct == 0:
            score += 0.1
        if split_correct == 0:
            score += 0.1
        if value_correct == 1:
            score += 0.1
        ERR_SCORE = 0.0
        score += 0.7 * (correct_num / len(truth)) - error_num * ERR_SCORE
        return score
        
    def process_menu(answer, truth_json):
        score = 0.0
        # extract truth
        truth = json.loads(truth_json)
        truth_set = set()
        for t in truth:
            truth_set.add(t.lower())
        correct_num = 0
        error_num = 0
        # define format score
        bracket_correct = 0 # score 0.1
        # extract solution
        answers = answer.split("\n")
        for ans in answers:
            if len(ans) == 0:
                continue
            if ans[0] != '[' or ans[-1] != ']':
                bracket_correct = -1
            else:
                ans = ans[1:-1]
            if ans.lower() in truth_set:
                correct_num += 1
                truth_set.remove(ans.lower())
            else:
                error_num += 1
        # add to score
        if bracket_correct == 0:
            score += 0.1
        ERR_SCORE = 0.0
        score += 0.9 * (correct_num / len(truth)) - error_num * ERR_SCORE
        return score
    
    def process_choice(answer, truth):
        score = 0.0
        # extract truth
        truth = truth.lower()
        # define format score
        bracket_correct = 0 # score 0.2
        # extract solution
        answer.strip()
        if len(answer) == 0:
            return .0
        if answer[0] != '[' or answer[-1] != ']':
            bracket_correct = -1
        else:
            answer = answer[1:-1]
        # add to score
        if bracket_correct == 0:
            score += 0.2
        if answer.lower() == truth:
            score += 0.8
        return score

    def process_value(answer, truth):
        score = 0.0
        # extract truth
        truth = truth.split("\n")
        truth_set = set()
        for t in truth:
            truth_set.add(t.lower())
        correct_num = 0
        # extract answers
        answers = answer.split("\n")
        for answer in answers:
            if answer.lower() in truth_set:
                correct_num += 1
        # add to score
        score += correct_num / len(truth)
        return score

    process_ty_dict = {
        "Bool": process_bool,
        "Menu": process_menu,
        "Choice": process_choice,
        "Value": process_value
    }

    # extract ground truth
    qa_type, answer_json = ground_truth.split("\t") # qa_type is on of ["Bool", "Menu", "Choice", "Value"]
    return process_ty_dict[qa_type](solution, answer_json)



def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None

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

def compute_score_answer(solution_str, ground_truth):
    if solution_str is None:
        return 0.0
    
    try:
        # Extract answer from <answer> tags
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)

        answer_reward = 0.0
        
        if answer is not None:
            # Check for exact match within <answer>
            # if em_check(answer, ground_truth):
            #     answer_reward = 1.0
            # # Check for substring match within <answer>
            # elif subem_check(answer, ground_truth):
            #     answer_reward = 0.5
            answer_reward = content_check(answer, ground_truth)
            if answer_reward > 1:
                log_path = Path("outputs") / "solution.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a+") as f:
                    f.write("SOLUTION\n" + solution_str + "\nTRUTH\n" + ground_truth +"\nANSWER REWARD\n" + str(answer_reward) + "\n\n\n")
        
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        print(traceback.format_exc())
        return 0.0
    
    return answer_reward

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
