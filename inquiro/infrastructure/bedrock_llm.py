"""BedrockLLM provider for Inquiro -- AWS Bedrock Claude integration 🌩️.

Provides a BaseLLM implementation that uses the ``anthropic.AnthropicBedrock``
SDK to call Claude models deployed on AWS Bedrock.  Because EvoMaster's
``LLMConfig`` does not carry AWS-specific fields, a factory function
``create_bedrock_llm`` is provided to construct the instance from a raw
provider config dict.

Example::

    provider_config = {
        "provider": "bedrock",
        "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "aws_region": "us-east-1",
        "aws_access_key_id": "${AWS_ACCESS_KEY_ID}",
        "aws_secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    llm = create_bedrock_llm(provider_config)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from evomaster.utils.llm import BaseLLM, LLMConfig, LLMResponse
from evomaster.utils.types import FunctionCall, ToolCall

logger = logging.getLogger(__name__)


class BedrockLLM(BaseLLM):
    """AWS Bedrock Claude LLM provider 🌩️.

    Wraps ``anthropic.AnthropicBedrock`` behind EvoMaster's ``BaseLLM``
    interface so that the rest of Inquiro can treat it as any other LLM.

    Attributes:
        aws_access_key_id: AWS access key for Bedrock authentication.
        aws_secret_access_key: AWS secret key for Bedrock authentication.
        aws_region: AWS region where the Bedrock endpoint lives.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        output_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize BedrockLLM with AWS credentials 🔧.

        Args:
            config: EvoMaster LLMConfig (provider is set to "bedrock").
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            aws_region: AWS region (e.g., "us-east-1").
            output_config: Optional output display config passed to BaseLLM.
        """
        # ✨ Store AWS params before super().__init__ because _setup() needs them.
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        super().__init__(config, output_config=output_config)

    def _setup(self) -> None:
        """Create the AnthropicBedrock client 🏗️."""
        try:
            from anthropic import AnthropicBedrock
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic[bedrock]"
            )

        self.client = AnthropicBedrock(
            aws_access_key=self.aws_access_key_id,
            aws_secret_key=self.aws_secret_access_key,
            aws_region=self.aws_region,
        )
        logger.info(
            "🌩️ BedrockLLM initialized: model=%s, region=%s",
            self.config.model,
            self.aws_region,
        )

    # ------------------------------------------------------------------
    # Core LLM call
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Anthropic Messages API via Bedrock (synchronous) 🚀.

        The method performs three conversions before calling the API:
        1. Extracts the system message (Anthropic requires it separate).
        2. Converts OpenAI-format messages to Anthropic format
           (tool role -> user/tool_result, assistant tool_calls -> content blocks).
        3. Converts tools from EvoMaster (OpenAI) format to Anthropic format.

        Args:
            messages: Message list in EvoMaster API format.
            tools: Tool specs in EvoMaster (OpenAI) format, or None.
            **kwargs: Overrides for temperature, max_tokens, timeout.

        Returns:
            LLMResponse with content, tool_calls, usage, etc.
        """
        # 📝 Separate system message and convert conversation messages.
        system_message: str | None = None
        conversation_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                conversation_messages.append(msg)

        # 🔄 Convert OpenAI format to Anthropic format
        conversation_messages = self._convert_messages_to_anthropic(
            conversation_messages
        )

        # 🏗️ Build request parameters.
        request_params: dict[str, Any] = {
            "model": self.config.model,
            "messages": conversation_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 4096),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if system_message:
            request_params["system"] = system_message

        # ⚠️ Bedrock does not accept tools=None; omit the key entirely.
        if tools:
            request_params["tools"] = self._convert_tools_to_anthropic(tools)
            request_params["tool_choice"] = kwargs.get("tool_choice", {"type": "auto"})

        # 🚀 Synchronous call to Bedrock.
        response = self.client.messages.create(**request_params)

        # 🔄 Parse Anthropic response into LLMResponse.
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Format conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages_to_anthropic(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format 🔄.

        Handles three key conversions:
        1. Assistant messages with ``tool_calls`` -> content blocks with tool_use.
        2. Tool-role messages -> user-role messages with tool_result blocks.
        3. Consecutive tool_result messages are merged into a single user message.

        Args:
            messages: Messages in OpenAI / EvoMaster format.

        Returns:
            Messages in Anthropic API format.
        """
        result: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def _flush_tool_results() -> None:
            """Flush pending tool results as a single user message 📦."""
            if pending_tool_results:
                result.append(
                    {
                        "role": "user",
                        "content": list(pending_tool_results),
                    }
                )
                pending_tool_results.clear()

        for msg in messages:
            role = msg.get("role", "")

            if role == "assistant":
                _flush_tool_results()
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # 🔄 Convert assistant tool_calls to Anthropic content blocks
                    content_blocks: list[dict[str, Any]] = []
                    text = msg.get("content")
                    if text:
                        content_blocks.append({"type": "text", "text": text})
                    for tc in tool_calls:
                        func = tc.get("function", tc)
                        args_str = func.get("arguments", "{}")
                        try:
                            input_data = (
                                json.loads(args_str)
                                if isinstance(args_str, str)
                                else args_str
                            )
                        except json.JSONDecodeError:
                            input_data = {"raw": args_str}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "input": input_data,
                            }
                        )
                    result.append(
                        {
                            "role": "assistant",
                            "content": content_blocks,
                        }
                    )
                else:
                    # ✨ Plain assistant text message
                    result.append(
                        {
                            "role": "assistant",
                            "content": msg.get("content", ""),
                        }
                    )

            elif role == "tool":
                # 🔄 Convert tool response to tool_result content block
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                )

            elif role == "user":
                _flush_tool_results()
                result.append(
                    {
                        "role": "user",
                        "content": msg.get("content", ""),
                    }
                )

        _flush_tool_results()
        return result

    @staticmethod
    def _convert_tools_to_anthropic(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert EvoMaster (OpenAI) tool specs to Anthropic format 🔄.

        EvoMaster ToolSpec (OpenAI style)::

            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": { ... },
                    "strict": null
                }
            }

        Anthropic format::

            {
                "name": "search",
                "description": "Search the web",
                "input_schema": { ... }
            }

        Args:
            tools: List of tool dicts in OpenAI / EvoMaster format.

        Returns:
            List of tool dicts in Anthropic format.
        """
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )
        return anthropic_tools

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Convert Anthropic Messages response to EvoMaster LLMResponse 🔄.

        Args:
            response: Raw response from ``client.messages.create()``.

        Returns:
            Populated LLMResponse.
        """
        content_text: str | None = None
        tool_calls: list[ToolCall] | None = None

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                ),
            },
            meta={
                "model": response.model,
                "response_id": response.id,
                "provider": "bedrock",
            },
        )


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------


def create_bedrock_llm(
    provider_config: dict[str, Any],
    output_config: dict[str, Any] | None = None,
) -> BedrockLLM:
    """Factory: build a BedrockLLM from a raw provider config dict 🏭.

    Since ``LLMConfig.provider`` is a Literal that does not include
    ``"bedrock"``, we create a minimal config with ``provider="anthropic"``
    (closest match) and store the AWS-specific fields separately.

    Args:
        provider_config: Dict typically loaded from ``llm_providers.yaml``::

            {
                "provider": "bedrock",
                "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "aws_region": "us-east-1",
                "aws_access_key_id": "...",
                "aws_secret_access_key": "...",
                "temperature": 0.7,
                "max_tokens": 4096,
            }

        output_config: Optional output display config.

    Returns:
        Ready-to-use BedrockLLM instance.

    Raises:
        ValueError: If required AWS credentials are missing.
    """
    aws_access_key_id = provider_config.get("aws_access_key_id", "")
    aws_secret_access_key = provider_config.get("aws_secret_access_key", "")
    aws_region = provider_config.get("aws_region", "us-east-1")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "Bedrock provider requires 'aws_access_key_id' and "
            "'aws_secret_access_key' in provider config"
        )

    # 📝 Create a minimal LLMConfig using "anthropic" as the provider
    # literal (closest match; the actual client is AnthropicBedrock).
    llm_config = LLMConfig(
        provider="anthropic",
        model=provider_config.get("model_id", ""),
        api_key="bedrock",  # ✨ Placeholder; actual auth uses AWS keys.
        temperature=provider_config.get("temperature", 0.7),
        max_tokens=provider_config.get("max_tokens"),
        timeout=provider_config.get("timeout", 300),
        max_retries=provider_config.get("max_retries", 3),
    )

    return BedrockLLM(
        config=llm_config,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        output_config=output_config,
    )
