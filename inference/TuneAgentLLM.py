import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ray
import re
import traceback
import torch
import numpy as np

import logging

from platform import processor
from hydra import compose, initialize
from omegaconf import OmegaConf

from tuneagent.src.fsdp_workers import ActorRolloutRefWorker
from tuneagent.llm_agent.generation import ToolGenerationManager, ToolGenerationConfig
from tuneagent.tool import ToolEnv
from tuneagent.tool.tools import _default_tools
from verl import DataProto
from verl.utils import hf_tokenizer
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs

class ChatContext:
    def __init__(self, target, config_path, config_name):
        self.init_prompt()
        self.init_target(target)
        self.init_model(config_path, config_name)
        
        # init logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("LLM.log", mode="w"))
        self.logger.propagate = False
        self.logger.info(target)
    
    def init_model(self, config_path, config_name):
        # init hydra config
        with initialize(config_path=config_path):
            self.cfg = compose(config_name)
        OmegaConf.resolve(self.cfg)

        if not ray.is_initialized():
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true'}})
        
        # 1. Initialize Tokenizer
        print("Initializing tokenizer...")
        self.tokenizer = hf_tokenizer(self.cfg.actor_rollout_ref.model.path, trust_remote_code=True)
        
        # 2. Setup Tool Environment
        print("Setting up tool environment...")
        tools = _default_tools(self.cfg.tool.env)
        self.env = ToolEnv(tools=tools, max_turns=self.cfg.tool.max_turns)

        # 3. Initialize the Actor/Rollout Worker Group directly
        print("Initializing Ray worker group...")
        # We only need the ActorRolloutRefWorker for inference.
        resource_pool = RayResourcePool(
                process_on_nodes=[self.cfg.trainer.n_gpus_per_node] * self.cfg.trainer.nnodes,
                use_gpu=True
        )
        actor_rollout_cls = RayClassWithInitArgs(
                cls=ActorRolloutRefWorker,
                config=self.cfg.actor_rollout_ref,
                role='actor_rollout' # Run in a mode that supports generation
        )
        actor_rollout_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_rollout_cls)
        actor_rollout_wg.init_model()
        
        # 4. Load the checkpoint into the worker group
        checkpoint_path = self.cfg.trainer.resume_mode
        print(f"Attempting to load checkpoint from: {checkpoint_path}")
        if checkpoint_path in (None, "", "auto", "disable"):
                raise FileNotFoundError(
                    "Inference requires a TuneAgent checkpoint. Set "
                    "trainer.resume_mode=/path/to/checkpoint in the Hydra config or "
                    "as a command-line override."
                )
        if not os.path.exists(os.path.join(checkpoint_path, 'actor')):
                raise FileNotFoundError(f"Checkpoint not found or 'actor' subdirectory missing in {checkpoint_path}")
        actor_checkpoint_path = os.path.join(checkpoint_path, 'actor')
        actor_rollout_wg.load_checkpoint(actor_checkpoint_path)
        print("Checkpoint loaded successfully!")

        # 5. Setup the Generation Manager
        gen_config = ToolGenerationConfig(
                max_turns=self.cfg.tool.max_turns,
                max_prompt_length=self.cfg.data.max_prompt_length,
                max_response_length=self.cfg.data.max_response_length,
                max_response_length_single_turn=self.cfg.data.max_response_length_single_turn,
                max_tool_response_length=self.cfg.data.max_tool_response_length,
                num_gpus=self.cfg.trainer.n_gpus_per_node,
                use_batch_tool_calls=self.cfg.tool.use_batch_tool_calls,
                tool_call_start=self.cfg.tool.tool_call_start,
                tool_call_end=self.cfg.tool.tool_call_end,
                tool_response_start=self.cfg.tool.tool_response_start,
                tool_response_end=self.cfg.tool.tool_response_end,
                tool_custom_response_template=self.cfg.tool.tool_custom_response_template,
        )
        self.generation_manager = ToolGenerationManager(
                tokenizer=self.tokenizer,
                processor=processor,
                actor_rollout_wg=actor_rollout_wg,
                config=gen_config,
                is_validation=True,
        )
        
    def init_target(self, target):
        self.target = target

    def init_prompt(self):
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
            "Value": instruction_header_value,
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
            "Value": instruction_tail_value,
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
            "Value": instruction_example_value,
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

        def get_prompt(qa_type):
            return (
                instruction_header[qa_type]
                + instruction_mid
                + instruction_tail[qa_type]
                + instruction_example[qa_type]
                + instruction_format
                + "Here are the given configs:\n{}"
            )

        self.menu_prompt = get_prompt("Menu")
        self.bool_prompt = get_prompt("Bool")
        self.choice_prompt = get_prompt("Choice")
        self.value_prompt = get_prompt("Value")

    def chat(self, content):
        prompt_with_template = self.tokenizer.apply_chat_template(
            content,
            tools=self.env.tool_desc,
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids = torch.tensor([self.tokenizer.encode(prompt_with_template)])
        
        # Create a DataProto object for the generation manager
        batch_data = {
            'input_ids': input_ids,
            'attention_mask': torch.ones_like(input_ids),
            'position_ids': torch.arange(0, input_ids.shape[1]).unsqueeze(0),
            'raw_prompt_ids': np.array([self.tokenizer.encode(prompt_with_template, add_special_tokens=False)], dtype=object)
        }
        gen_batch = DataProto.from_single_dict(batch_data)
        gen_batch.meta_info = {'do_sample': False}
        # initial_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
        # initial_input_ids = gen_batch.batch['input_ids'].clone()
        inference_envs = [self.env.copy()]

        # Generate the full trajectory
        output_gen_batch = self.generation_manager.run_llm_loop(
            gen_batch,
            envs=inference_envs,
            # initial_input_ids=initial_input_ids
        )
        
        # Decode and print the output
        response_tokens = output_gen_batch.batch['responses'][0]
        # full_trajectory_tokens = torch.cat([input_ids[0], response_tokens], dim=0)
        full_trajectory_tokens = torch.cat([response_tokens], dim=0)
        decoded_full = self.tokenizer.decode(full_trajectory_tokens, skip_special_tokens=False)

        # extract answer
        try:
            assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', decoded_full, re.DOTALL)
            solution_str = assistant_blocks[-1]
            # Extract the answer from the solution string.
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.search(answer_pattern, solution_str, re.DOTALL)
            
            if match:
                return match.group(1).strip()
            print(f"[DEBUG] Cannot find answer block, answer is {decoded_full}")
            return None
        except Exception as e:
            print(f"[DEBUG] Error while extracting answer: {decoded_full}, error is: {e}")
            print(traceback.format_exc())
            return None

    def ask_menu(self, content: str) -> list[str]:
        conversation = [
            {
                "role": "user",
                "content": self.menu_prompt.format(self.target, content),
            }
        ]
        answer = self.chat(conversation)
        if answer is None:
            return []
        answers = answer.split("\n")
        def remove_bracket(s):
            if s[0] == '[' and s[-1] == ']':
                return s[1:-1]
        new_answers = list(map(remove_bracket, answers))
        return new_answers

    def ask_bool(self, content: str) -> dict[str:int]:
        conversation = [
            {
                "role": "user",
                "content": self.bool_prompt.format(self.target, content),
            }
        ]
        answer = self.chat(conversation)
        if answer is None:
            return {}
        answers = answer.split("\n")
        def remove_bracket(s):
            if s[0] == '[' and s[-1] == ']':
                return s[1:-1]
        new_answers = list(map(remove_bracket, answers))
        value_map = {}
        for ans in new_answers:
            if ans is None:
                continue
            l = ans.split("-")
            res = l[1].strip()
            if res == "increase":
                value_map[l[0].strip()] = 2
            elif res == "decrease":
                value_map[l[0].strip()] = 0
        print(value_map)
        return value_map


    def ask_choice(self, content: str) -> dict[str:int]:
        conversation = [
            {
                "role": "user",
                "content": self.choice_prompt.format(self.target, content),
            }
        ]
        answer = self.chat(conversation)
        if answer is None:
            return ""
        if answer[0] == '[' and answer[-1] == ']':
            answer = answer[1:-1]
        return answer

    def ask_value(self, content: str) -> dict[str:int]:
        conversation = [
            {
                "role": "user",
                "content": self.value_prompt.format(self.target, content),
            }
        ]
        answer = self.chat(conversation)
        if answer is None:
            return []
        answers = answer.split("\n")
        results = []
        for ans in answers:
            match = re.search(r'^(.*?)\((.*?)\)', answer)
            if match is None:
                continue
            results.append((match.group(1).strip(), match.group(2).strip()))
        return results


if __name__ == "__main__":
    ctx = ChatContext("I want to improve the performance of unixbench", './tuneagent/src/config', 'agent_trainer_inference')
    # ctx.ask_menu("General setup\nProcessor type and features\nPower management and ACPI options\nBus options (PCI etc.)\nBinary Emulations\nGeneral architecture-dependent options\nExecutable file formats\nMemory Management options\nDevice Drivers\nFile systems\nSecurity options\nCryptographic API (CRYPTO)\nLibrary routines\nKernel hacking\nMitigations for speculative execution vulnerabilities (SPECULATION_MITIGATIONS)\n15 Virtualization (VIRTUALIZATION)\nEnable loadable module support (MODULES)\nEnable the block layer (BLOCK)\nNetworking support (NET)\n")
    # ctx.ask_bool("Sign modules with SHA-256 (MODULE_SIG_SHA256)\nSign modules with SHA-384 (MODULE_SIG_SHA384)\nSign modules with SHA-512 (MODULE_SIG_SHA512)\nSign modules with SHA3-256 (MODULE_SIG_SHA3_256)\nSign modules with SHA3-384 (MODULE_SIG_SHA3_384)\nSign modules with SHA3-512 (MODULE_SIG_SHA3_512)\n")
    ctx.ask_value("Physical address where the kernel is loaded (0x1000000)\nAlignment value to which kernel should be aligned (0x200000)")
