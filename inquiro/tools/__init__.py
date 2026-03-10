"""Inquiro Tools — custom agent tools for research & synthesis 🛠️.

Provides Inquiro-specific tools that extend EvoMaster's BaseTool:
    InquiroFinishTool: Schema-enforcing finish tool 📋
    RequestResearchTool: Deep-dive research trigger for SynthesisAgent 🔬
    MCPToolWrapper: Generic MCP-to-BaseTool bridge 🔧
"""

from inquiro.tools.finish_tool import (
    InquiroFinishTool,
    InquiroFinishToolParams,
)
from inquiro.tools.mcp_tool_wrapper import (
    MCPToolWrapper,
)
from inquiro.tools.request_research_tool import (
    RequestResearchTool,
    RequestResearchToolParams,
)

__all__ = [
    "InquiroFinishTool",
    "InquiroFinishToolParams",
    "MCPToolWrapper",
    "RequestResearchTool",
    "RequestResearchToolParams",
]
