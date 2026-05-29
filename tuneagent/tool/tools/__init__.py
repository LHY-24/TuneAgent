"""
Tools for TuneAgent
"""

# Import existing tools
from tuneagent.tool.tools.search_tool import SearchTool
from tuneagent.tool.tools.calculator_tool import CalculatorTool
from tuneagent.tool.tools.wiki_search_tool import WikiSearchTool
from tuneagent.tool.tools.kernel_knowledge_tool import GenKnowledgeTool, GenConfigsKnowledgeTool

__all__ = [
    'SearchTool',
    'CalculatorTool', 
    'WikiSearchTool',
    'GenKnowledgeTool',
    'GenConfigsKnowledgeTool',
]

def _default_tools(env):
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'knowledge_base':
        return [GenKnowledgeTool()]
    elif env == 'config_analysis':
        return [GenConfigsKnowledgeTool()]
    elif env == 'knowledge_tools':
        # Return both LightRAG-based tools together
        # return [GenKnowledgeTool(), GenConfigsKnowledgeTool()]
        return [GenConfigsKnowledgeTool()]
    elif env == 'all':
        # Return all available tools
        return [
            SearchTool(),
            CalculatorTool(),
            WikiSearchTool(),
            GenKnowledgeTool(),
            GenConfigsKnowledgeTool()
        ]
    else:
        return []
