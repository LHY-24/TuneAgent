"""
Tool generation manager for LLM agents
"""

import torch
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .tensor_helper import TensorHelper, TensorConfig
from tuneagent.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_prompt_length: int 
    max_response_length: int
    max_response_length_single_turn: int
    max_tool_response_length: int
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tool_custom_response_template: str = ""
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        processor,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp + self.tokenizer.eos_token, False  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            
            return resp + self.tokenizer.eos_token, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        return responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        tool_response_images = [None] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]['content']  # Extract observation from (observation, reward, done, info)
            tool_response_images[i] = result[0]['image']
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses, tool_response_images
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        tool_response_images = [None] * len(response_strs)
        
        if not active_envs:
            return tool_responses, tool_response_images
            
        # Use the independent step_batch function for active environments
        batch_results = step_batch(active_envs, active_responses)
        
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
                tool_response_images[idx] = None
            else:
                tool_response = result[0]['content']  # Extract observation from (observation, reward, done, info)
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
                tool_response_images[idx] = result[0]['image']
        return tool_responses, tool_response_images
 
    def _update_rolling_state(self, rollings, cur_responses: List[str], 
                            tool_responses: List[str], tool_responses_images: List[List[Image.Image]]) -> Dict:
        """Update rolling state with new responses and observations.
        rollings : last llm input DataProto
        cur_responses: llm output action 
        tool_responses: tool response text
        tool_responses_images: tool response image 
        """

        is_multi_modal = "multi_modal_data" in rollings.non_tensor_batch.keys()

        row_dict_list = []
        responses = []
        raw_prompts = []
        
        for i, (tool_response, cur_response, tool_responses_image) in enumerate(zip(tool_responses, cur_responses, tool_responses_images)):
            row_dict={}
            if is_multi_modal and '<image>' in tool_response and tool_responses_image is not None:
                assert len(tool_responses_image) == tool_response.count('<image>'), f"[WARNING] TOOL RESPONSE IMAGE NUMBER NOT MATCH, {len(tool_responses_image)} != {tool_response.count('<image>')} for {cur_response}"
                raw_prompts.append(cur_response + tool_response.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>'))
                row_dict['multi_modal_data'] = {'image': tool_responses_image}
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index = 0
                    while '<image>' in tool_response:
                        tool_response = tool_response.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1

                    tool_response = tool_response.replace('<|placeholder|>', self.processor.image_token)

            else:
                raw_prompts.append(cur_response + tool_response)
            responses.append(cur_response + tool_response)
            row_dict_list.append(row_dict)

        responses_ids = self._batch_tokenize(responses)

        if "responses" not in rollings.batch.keys():
            rollings.batch['responses'] = responses_ids[:, :self.config.max_response_length_single_turn]
        else:
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                rollings.batch['responses'],
                responses_ids[:, :self.config.max_response_length_single_turn]
            ], pad_to_left=False)

        rollings.batch['responses'] = rollings.batch['responses'][:, :self.config.max_response_length]

        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        if is_multi_modal:
            multi_modal_data = rollings.non_tensor_batch['multi_modal_data']
            multi_modal_inputs = rollings.non_tensor_batch['multi_modal_inputs']

            new_multi_modal_data = []
            new_multi_modal_inputs = []

            for row_dict, multi_modal_data_, multi_modal_inputs_ in zip(row_dict_list, multi_modal_data, multi_modal_inputs):
                if 'multi_modal_data' in row_dict.keys():
                    new_multi_modal_data.append({"image":multi_modal_data_['image'] + row_dict['multi_modal_data']['image']})
                else:
                    new_multi_modal_data.append({"image":multi_modal_data_['image']})
                if 'multi_modal_inputs' in row_dict.keys():
                    new_multi_modal_inputs.append({key: torch.cat((val,row_dict['multi_modal_inputs'][key]),dim=0) for key, val in multi_modal_inputs_.items()})
                else:
                    new_multi_modal_inputs.append({key: val for key, val in multi_modal_inputs_.items()})

            rollings.non_tensor_batch['multi_modal_data'] = np.array(new_multi_modal_data, dtype=object)
            rollings.non_tensor_batch['multi_modal_inputs'] = np.array(new_multi_modal_inputs, dtype=object)

            from verl.models.transformers.qwen2_vl import get_rope_index
            new_postion_ids = []
            for i in range(len(new_multi_modal_data)):
                new_postion_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=new_input_ids[i],
                    image_grid_thw=new_multi_modal_inputs[i]['image_grid_thw'],
                    attention_mask=new_attention_mask[i],
                ))

            new_position_ids = torch.stack(new_postion_ids, dim=0)
        else:
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        rollings.batch['input_ids'] = new_input_ids
        rollings.batch['position_ids'] = new_position_ids
        rollings.batch['attention_mask'] = new_attention_mask

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []

        for raw_prompt_id, raw_prompt in zip(raw_prompt_ids, raw_prompts):
            if len(raw_prompt) > 0:
                new_raw_prompt_id = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                if len(new_raw_prompt_id) > self.config.max_response_length_single_turn:
                    print(f"[WARNING] RESPONSE TOO LONG ({len(new_raw_prompt_id)}/{self.config.max_response_length_single_turn}), TRUNCATED: {raw_prompt}")
                    new_raw_prompt_id = new_raw_prompt_id[:self.config.max_response_length_single_turn]
                # Create a new list instead of extending the existing one
                new_raw_prompt_ids.append(raw_prompt_id + new_raw_prompt_id)
            else:
                new_raw_prompt_ids.append(raw_prompt_id)

        rollings.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

        return rollings
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        prompts = gen_batch.batch['input_ids'][:, -self.config.max_prompt_length:].clone()

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break

            # Check if any sequence exceeds max length
            effective_len = rollings.batch['attention_mask'].sum(dim=1)
            length_exceeded = effective_len > self.config.max_prompt_length

            if length_exceeded.sum() > 0:
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                active_mask[length_exceeded] = 0

            raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
            length_exceeded = [len(prompt_id) > self.config.max_response_length for prompt_id in raw_prompt_ids]
            if any(length_exceeded):
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                for prompt_id, length_exceeded_ in zip(raw_prompt_ids, length_exceeded):
                    if length_exceeded_:
                        print(f"[DEBUG] LENGTH {len(prompt_id)}: {self.tokenizer.decode(prompt_id)}")
                active_mask[length_exceeded] = 0
            
            if not active_mask.sum():
                print("[WARNING] NO ACTIVE SEQUENCES")
                break
            
            if hasattr(rollings, 'non_tensor_batch') and rollings.non_tensor_batch:
                # Create active batch with tensor data
                rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    non_tensors={
                        k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()
                    }
                )
            else:
                rollings_active = DataProto.from_dict(
                    batch={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                )

            # rollings_active.batch = self.tensor_fn.cut_to_effective_len(
            #     rollings_active.batch,
            #     keys=['input_ids', 'attention_mask', 'position_ids']
            # )

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_str = self.tensor_fn._example_level_pad(responses_str, active_mask)
          
            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses, tool_responses_images = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses, tool_responses_images = self._execute_tool_calls(responses_str, envs, active_mask)

            active_num_list.append(active_mask.sum().item())

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_str,
                tool_responses,
                tool_responses_images
            )
 
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # Compose final output
        final_output = {}
        final_output['turns'] = turns
        final_output['prompts'] = prompts
        final_output['responses'] = rollings.batch['responses']
        final_output['input_ids'] = torch.cat([
            prompts,
            rollings.batch['responses']
        ], dim=1)
        final_output['attention_mask'] = self.tensor_fn.create_attention_mask(final_output['input_ids'])
        if "multi_modal_data" in rollings.non_tensor_batch.keys():
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                position_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=final_output['input_ids'][i],
                    image_grid_thw=rollings.non_tensor_batch['multi_modal_inputs'][i]['image_grid_thw'],
                    attention_mask=final_output['attention_mask'][i],
                ))

            position_ids = torch.stack(position_ids, dim=0)
            final_output['position_ids'] = position_ids
        else:
            final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch = rollings.non_tensor_batch
        
        return final_output