# PATH: tuneagent/tool/tools/kernel_knowledge_tool.py


"""
Kernel knowledge tools implementation for knowledge generation and config evaluation
"""

import os
import json
from typing import Dict, List, Any, Optional

import traceback

# --- Tool Base Import ---
try:
    from tuneagent.tool.tool_base import Tool
except ImportError:
    raise ImportError("Could not import Tool base class from tuneagent.tool.tool_base")

# --- LightRAG Imports ---
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    import kconfiglib as klib
except ImportError:
    raise ImportError("Could not import LightRAG. Please install it: pip install lightrag")

# --- Configuration Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIGHTRAG_WORKING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../data/lightrag_knowledge_base"))
LIGHTRAG_WORKING_DIR = os.environ.get("LIGHTRAG_WORKING_DIR", DEFAULT_LIGHTRAG_WORKING_DIR)

DEFAULT_SEARCH_MODE = os.environ.get("LIGHTRAG_SEARCH_MODE", "hybrid")
DEFAULT_LLM_FUNC = os.environ.get("LIGHTRAG_LLM_FUNC", "gpt-4o-mini")
DEFAULT_GENERATE_KNOWLEDGE_FLAG = os.environ.get("LIGHTRAG_TOOL_ENABLED", "true").lower() == "true"


# --- Internal Core Logic Class ---
class _LightRAGCore:
    """
    Internal helper class to encapsulate LightRAG initialization and core methods.
    This is instantiated once by each tool that needs it.
    """
    def __init__(
        self,
        working_dir: str,
        gen_knowledge_flag: bool,
        search_mode: str,
        llm_model_func_name: str = "gpt-4o-mini",
    ):
        self.is_enabled = gen_knowledge_flag
        self.rag_instance = None

        if not self.is_enabled:
            print(f"[INFO] LightRAG Core generation DISABLED.")
            return  # Skip initialization if disabled

        if not os.path.isdir(working_dir):
            print(f"[ERROR] LightRAG working directory not found: {working_dir}")
            print(f"[WARN] LightRAG Core DISABLED due to missing directory.")
            self.is_enabled = False
            return

        # Select LLM function
        model_map = {
            "gpt-4o-mini": gpt_4o_mini_complete,
            "gpt-4o": gpt_4o_complete,
        }
        model_func = model_map.get(llm_model_func_name, gpt_4o_mini_complete)
        if llm_model_func_name not in model_map:
            print(f"[WARN] LLM function '{llm_model_func_name}' not recognized, defaulting to gpt-4o-mini.")

        print(f"[DEBUG] Initializing LightRAG Core...")
        print(f"[DEBUG]   Working Directory: {working_dir}")
        print(f"[DEBUG]   Search Mode: {search_mode}")
        print(f"[DEBUG]   LLM Function: {llm_model_func_name}")
        try:
            # Initialize LightRAG instance
            self.rag_instance = LightRAG(
                working_dir=working_dir,
                llm_model_func=model_func,
                log_level="ERROR",  # Keep logs clean unless debugging
            )
            self.search_mode = search_mode
            print(f"[DEBUG] LightRAG Core initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize LightRAG Core: {e}")
            self.is_enabled = False  # Disable if init fails
            self.rag_instance = None

    def gen_knowledge(self, prompt: str) -> str:
        """Generates knowledge for a given prompt."""
        if not self.is_enabled or self.rag_instance is None:
            return "Error: LightRAG Core is disabled or not initialized."
        try:
            print(f"[DEBUG] LightRAG Core querying with prompt: '{prompt[:100]}...'")
            result = self.rag_instance.query(prompt, QueryParam(mode=self.search_mode))
            print(f"[DEBUG] LightRAG Core query successful.")
            return str(result)  # Ensure string output
        except Exception as e:
            print(f"[ERROR] LightRAG Core gen_knowledge error: {e}")
            print(traceback.format_exc())
            return f"Error: Exception during LightRAG query: {type(e).__name__}"

    def gen_configs_knowledge(self, configs: List[str], target: str) -> str:
        """Generates knowledge about config impact on a target."""
        if not self.is_enabled or self.rag_instance is None:
            return "Error: LightRAG Core is disabled or not initialized."

        # Basic input validation
        if not isinstance(configs, list) or not target:
            return "Error: Invalid input for gen_configs_knowledge (configs must be a list, target must be non-empty)."

        prompt = f"Of these configs listed below, which ones may affect the target?: {target}\n---\n"
        prompt += "\n".join([f"- {cfg}" for cfg in configs])
        prompt += "\n---"
        # Use the gen_knowledge method to perform the actual query
        return self.gen_knowledge(prompt)


# --- Tool Definition 1: General Knowledge Query ---
class GenKnowledgeTool(Tool):
    """
    Tool to query the internal LightRAG knowledge base with a text prompt.
    Use this for retrieving specific, grounded information based on the internal data.
    """
    def __init__(
        self,
        working_dir: str = LIGHTRAG_WORKING_DIR,
        search_mode: str = DEFAULT_SEARCH_MODE,
        llm_model_func: str = DEFAULT_LLM_FUNC,
        enabled: bool = DEFAULT_GENERATE_KNOWLEDGE_FLAG
    ):
        name = "query_knowledge_base"
        description = (
            "This tool can be used to generate detailed knowledge for given configs."
            # "Retrieves specific, grounded knowledge for a given text prompt using the internal LightRAG system. "
            # "Use this for detailed info from internal documents (e.g., OS kernel details), not for general web search."
        )
        # Parameters for the agent to provide
        parameters = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The specific text query or prompt to retrieve knowledge about."
                }
            },
            "required": ["prompt"]
        }
        super().__init__(name, description, parameters)

        # Initialize the backend ONCE
        self.rag_core = _LightRAGCore(
            working_dir=working_dir,
            gen_knowledge_flag=enabled,
            search_mode=search_mode,
            llm_model_func_name=llm_model_func,
        )

    def execute(self, args: Dict) -> str:
        """Single execution - delegates to batch_execute"""
        results = self.batch_execute([args])
        return results[0] if results else "Error: Failed to execute query_knowledge_base tool"

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """Batch execution for multiple knowledge queries"""
        results = []
        for args in args_list:
            prompt = args.get("prompt", "").strip()
            if not prompt:
                results.append("Error: Missing required parameter 'prompt'.")
                continue
            # Call the method on the pre-initialized core instance
            result = self.rag_core.gen_knowledge(prompt)
            results.append(result)
        return results

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for the knowledge generator action.

        Args:
            args: Tool parameters for a single call (contains 'prompt').
            result: The string result returned by the tool for that call.

        Returns:
            Reward value (e.g., 0.1 for success, 0.0 for error).
        """
        # Check if the result string indicates an error condition
        if isinstance(result, str) and result.startswith("Error:"):
             # Optionally, different negative rewards for different errors
            return 0.0 # Treat errors as neutral or slightly negative if needed
        elif isinstance(result, str) and result: # Check if result is a non-empty string and not an error
            return 0.1  # Successful execution reported
        else:
            # Handle unexpected result types or empty strings if necessary
            return 0.0


# --- Tool Definition 2: Config Impact Analysis ---
class GenConfigsKnowledgeTool(Tool):
    """
    Tool to analyze which configurations from a list might affect a specific target,
    using the internal LightRAG knowledge base.
    """
    def __init__(
        self,
        working_dir: str = LIGHTRAG_WORKING_DIR,
        search_mode: str = DEFAULT_SEARCH_MODE,
        llm_model_func: str = DEFAULT_LLM_FUNC,
        enabled: bool = DEFAULT_GENERATE_KNOWLEDGE_FLAG
    ):
        name = "analyze_config_impact"
        description = (
            "This tool can be used to generate detailed knowledge for given configs."
            # "Given a list of configuration item names/descriptions and an optimization target description, "
            # "uses the internal LightRAG system to identify which configurations might affect the target."
        )
        # Parameters for the agent to provide
        parameters = {
            "type": "object",
            "properties": {
                "configs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of config names to be processed. You should use all of them as the parameters of the tool."
                },
                "target": {
                    "type": "string",
                    "description": "A description of the optimization target or goal (e.g., 'improve network throughput', 'reduce memory usage')."
                },
            },
            "required": ["configs", "target"]
        }
        super().__init__(name, description, parameters)

        # Initialize the backend ONCE
        self.rag_core = _LightRAGCore(
            working_dir=working_dir,
            gen_knowledge_flag=enabled,
            search_mode=search_mode,
            llm_model_func_name=llm_model_func,
        )

    def execute(self, args: Dict) -> str:
        """Single execution - delegates to batch_execute"""
        results = self.batch_execute([args])
        return results[0] if results else "Error: Failed to execute analyze_config_impact tool"

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        """Batch execution for multiple config analyses"""
        results = []
        for args in args_list:
            configs = args.get("configs")
            target = args.get("target", "").strip()

            # Validate required arguments for this specific tool call
            if not isinstance(configs, list) or not target:
                error_msg = "Error: Missing or invalid required parameters 'configs' (must be a list of strings) or 'target' (must be a non-empty string)."
                results.append(error_msg)
                continue

            # Call the method on the pre-initialized core instance
            result = self.rag_core.gen_configs_knowledge(configs, target)
            results.append(result)
        return results

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for the analyze config impact action.

        Args:
            args: Tool parameters for a single call (contains 'configs', 'target').
            result: The string result returned by the tool for that call.

        Returns:
            Reward value (e.g., 0.1 for success, 0.0 for error).
        """
        # Check if the result string indicates an error condition
        if isinstance(result, str) and result.startswith("Error:"):
             # Optionally, different negative rewards for different errors
            return 0.0 # Treat errors as neutral or slightly negative if needed
        elif isinstance(result, str) and result: # Check if result is a non-empty string and not an error
            return 0.1  # Successful execution reported
        else:
            # Handle unexpected result types or empty strings if necessary
            return 0.0
