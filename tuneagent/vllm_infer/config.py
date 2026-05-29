"""
Configuration parameters for the VLLM inference
"""

# Environment and API settings
ENV = "search"
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "agent"

# Model inference parameters
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 512
REPETITION_PENALTY = 1.05

# Instruction template
INSTRUCTION_FOLLOWING = """Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
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