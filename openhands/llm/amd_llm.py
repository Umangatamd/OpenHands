"""AMD LLM Gateway integration for OpenHands.

This module provides integration with AMD's internal LLM API Gateway,
supporting both GPT models (via Azure OpenAI) and Claude models (via Anthropic).

Supported models:
- GPT: gpt-5, gpt-5-codex, gpt-5.1
- Claude: claude-opus-4.5, claude-sonnet-4.5
"""

import copy
import os
import time
import warnings
from typing import Any, Callable, cast

import anthropic
import openai

from openhands.core.config import LLMConfig
from openhands.core.exceptions import LLMNoResponseError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics
from openhands.llm.fn_call_converter import (
    STOP_WORDS,
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)

from litellm.types.utils import ModelResponse, Usage, Choices, Message as LiteLLMMessage

__all__ = ['AmdLLM']

# Exceptions to retry on
AMD_LLM_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.APIStatusError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.APIStatusError,
    LLMNoResponseError,
)


class AmdLLM(LLM):
    """AMD LLM Gateway client for OpenHands.
    
    This class provides access to AMD's internal LLM API Gateway which supports
    GPT models (via Azure-style API) and Claude models (via Anthropic API).
    
    Authentication is done via the `Ocp-Apim-Subscription-Key` header.
    """

    def __init__(
        self,
        config: LLMConfig,
        service_id: str,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        # Initialize basic attributes first (before calling super)
        self.config: LLMConfig = copy.deepcopy(config)
        self.service_id = service_id
        self.metrics: Metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self.retry_listener = retry_listener
        self.cost_metric_supported: bool = True
        self._function_calling_active: bool = False  # AMD gateway doesn't support native function calling
        self.model_info = None
        self._tried_model_info = False
        
        # Get API key from config or environment
        self.api_key = (
            self.config.api_key.get_secret_value() if self.config.api_key 
            else os.getenv("AMD_LLM_API_KEY") 
            or os.getenv("LLM_GATEWAY_KEY")
        )
        
        if not self.api_key:
            raise ValueError(
                "AMD LLM API key not found. Set AMD_LLM_API_KEY or LLM_GATEWAY_KEY environment variable, "
                "or provide api_key in config."
            )
        
        # Get user name safely for headers
        try:
            self.user = os.getlogin()
        except OSError:
            self.user = os.getenv("USER", "unknown")
        
        # Extract model name (remove 'amd/' prefix if present)
        self.model_name = self.config.model.removeprefix('amd/')
        
        # Initialize the appropriate client based on model type
        self._init_client()
        
        logger.info(f"Initialized AMD LLM with model: {self.model_name}")

    def _init_client(self) -> None:
        """Initialize the appropriate client (OpenAI or Anthropic) based on model name."""
        if self._is_gpt_model():
            base_url = self.config.base_url or f"https://llm-api.amd.com/openai/{self.model_name}"
            self.client = openai.AzureOpenAI(
                api_key="dummy",  # Actual auth is via header
                api_version=self.config.api_version or "2023-10-16",
                base_url=base_url,
                default_headers={
                    "Ocp-Apim-Subscription-Key": self.api_key,
                },
            )
            self.client_type = "openai"
            logger.debug(f"Initialized OpenAI client for AMD with base_url: {base_url}")
        elif self._is_claude_model():
            base_url = self.config.base_url or "https://llm-api.amd.com/Anthropic"
            self.client = anthropic.Anthropic(
                api_key="dummy",  # Actual auth is via header
                base_url=base_url,
                default_headers={
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "user": self.user,
                    "anthropic-version": self.config.api_version or "2023-10-16",
                },
            )
            self.client_type = "anthropic"
            logger.debug(f"Initialized Anthropic client for AMD with base_url: {base_url}")
        else:
            raise ValueError(
                f"Unsupported AMD model: {self.model_name}. "
                "Supported models: gpt-5, gpt-5-codex, gpt-5.1, claude-opus-4.5, claude-sonnet-4.5"
            )

    def _is_gpt_model(self) -> bool:
        return "gpt" in self.model_name.lower()

    def _is_claude_model(self) -> bool:
        return "claude" in self.model_name.lower()

    @property
    def completion(self) -> Callable:
        """Return the completion function with retry logic."""
        return self._completion_with_retry

    def _completion_with_retry(self, *args, **kwargs) -> ModelResponse:
        """Wrapper for completion with retry logic and mock function calling."""
        
        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=AMD_LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
            retry_listener=self.retry_listener,
        )
        def _inner(*args, **kwargs) -> ModelResponse:
            return self._do_completion(*args, **kwargs)
        
        return _inner(*args, **kwargs)

    def _do_completion(self, messages: list | None = None, **kwargs) -> ModelResponse:
        """Execute the completion request with mock function calling support."""
        
        if messages is None:
            messages = kwargs.get('messages', [])
        
        # Convert Message objects to dict format if needed
        if messages and isinstance(messages[0], Message):
            messages = self.format_messages_for_llm(messages)
        
        # Ensure messages are dicts
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            elif hasattr(msg, 'model_dump'):
                formatted_messages.append(msg.model_dump())
            else:
                formatted_messages.append({
                    'role': getattr(msg, 'role', 'user'),
                    'content': str(getattr(msg, 'content', msg))
                })
        
        messages = formatted_messages
        
        # Handle mock function calling - convert tools to prompts
        mock_fncall_tools = None
        original_messages = copy.deepcopy(messages)
        
        if 'tools' in kwargs and kwargs['tools']:
            # Convert function calling messages to non-function calling format
            # This injects tool definitions into the prompt
            logger.debug(f"Converting {len(kwargs['tools'])} tools to prompt format")
            messages = convert_fncall_messages_to_non_fncall_messages(
                messages,
                kwargs['tools'],
                add_in_context_learning_example=True,
            )
            mock_fncall_tools = kwargs.pop('tools')
            kwargs.pop('tool_choice', None)  # Remove tool_choice when mocking
            
            # Add stop words
            kwargs['stop'] = STOP_WORDS
        
        # Log the prompt
        self.log_prompt(messages)
        
        start_time = time.time()
        
        # Make the actual API call
        if self.client_type == "openai":
            response = self._query_openai(messages, **kwargs)
        else:
            response = self._query_anthropic(messages, **kwargs)
        
        latency = time.time() - start_time
        logger.debug(f"LLM call took {latency:.2f}s")
        
        # Convert to ModelResponse format
        model_response = self._to_model_response(response)
        
        # If we mocked function calling, convert the response back to function call format
        if mock_fncall_tools:
            logger.debug("Converting response back to function call format")
            # Get the response message
            non_fncall_response_message = model_response.choices[0].message
            # Convert messages + response to function call format
            fn_call_messages_with_response = convert_non_fncall_messages_to_fncall_messages(
                messages + [{"role": "assistant", "content": non_fncall_response_message.content or ""}],
                mock_fncall_tools,
            )
            # Extract the converted response message (last item)
            fn_call_response_message = fn_call_messages_with_response[-1]
            if not isinstance(fn_call_response_message, LiteLLMMessage):
                fn_call_response_message = LiteLLMMessage(**fn_call_response_message)
            model_response.choices[0].message = fn_call_response_message
        
        # Log response
        self.log_response(model_response)
        
        # Update metrics
        self._update_metrics(model_response, latency)
        
        return model_response

    def _query_openai(self, messages: list[dict], **kwargs) -> Any:
        """Query GPT models via AMD's OpenAI-compatible API."""
        # AMD Responses API supported parameters
        supported_params = {
            "top_p", "frequency_penalty", "presence_penalty", 
            "stop", "stream", "n", "seed", "response_format", 
            "reasoning", "text"
        }
        
        # Filter parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items()
            if k in supported_params
        }
        
        # Add reasoning effort if configured
        if self.config.reasoning_effort:
            filtered_kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
        
        # Concatenate messages into a single prompt for the Responses API
        prompt = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
            for msg in messages
        ])
        
        # Call AMD Responses API
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            **filtered_kwargs,
        )
        
        return response

    def _query_anthropic(self, messages: list[dict], **kwargs) -> Any:
        """Query Claude models via AMD's Anthropic-compatible API."""
        # Anthropic API supported parameters
        supported_params = {
            "temperature", "max_tokens", "top_p", "top_k",
            "stop_sequences", "stream", "metadata", "system"
        }
        
        # Filter parameters - convert 'stop' to 'stop_sequences' for Anthropic
        filtered_kwargs = {}
        for k, v in kwargs.items():
            if k == 'stop':
                filtered_kwargs['stop_sequences'] = v
            elif k in supported_params:
                filtered_kwargs[k] = v
        
        # Set defaults
        if "temperature" not in filtered_kwargs:
            filtered_kwargs["temperature"] = self.config.temperature
        if "max_tokens" not in filtered_kwargs:
            filtered_kwargs["max_tokens"] = self.config.max_output_tokens or 4096
        
        # Convert messages format for Anthropic API
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle content that might be a list (for vision messages)
            if isinstance(content, list):
                content = " ".join([
                    c.get("text", "") if isinstance(c, dict) else str(c) 
                    for c in content
                ])
            
            if role == "system":
                system_message = content
            else:
                anthropic_role = "assistant" if role == "assistant" else "user"
                anthropic_messages.append({
                    "role": anthropic_role,
                    "content": content
                })
        
        # Add system message if present
        if system_message:
            filtered_kwargs["system"] = system_message
        
        # Call Anthropic API
        response = self.client.messages.create(
            model=self.model_name,
            messages=anthropic_messages,
            **filtered_kwargs
        )
        
        return response

    def _to_model_response(self, response: Any) -> ModelResponse:
        """Convert AMD API response to LiteLLM ModelResponse format."""
        if self.client_type == "openai":
            content = self._parse_openai_response(response)
            usage = Usage(
                prompt_tokens=getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
                completion_tokens=getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
                total_tokens=(
                    (getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0))
                    if response.usage else 0
                ),
            )
        else:
            content = self._parse_anthropic_response(response)
            usage = Usage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
            )
        
        # Create ModelResponse compatible with OpenHands
        model_response = ModelResponse(
            id=getattr(response, 'id', f"amd-{time.time()}"),
            choices=[
                Choices(
                    index=0,
                    message=LiteLLMMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            model=self.model_name,
            usage=usage,
        )
        
        return model_response

    def _parse_openai_response(self, response: Any) -> str:
        """Parse OpenAI/GPT response content."""
        content = ""
        try:
            out = response.output
            if out and hasattr(out[0], "content"):
                if out[0].content and out[0].content[0].type == "output_text":
                    content = out[0].content[0].text
            elif out and hasattr(out[-1], "content"):
                if out[-1].content and out[-1].content[0].type == "output_text":
                    content = out[-1].content[0].text
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI response content: {e}")
        return content

    def _parse_anthropic_response(self, response: Any) -> str:
        """Parse Anthropic/Claude response content."""
        content = ""
        try:
            if response.content:
                content_parts = []
                for block in response.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                content = "".join(content_parts)
        except Exception as e:
            logger.warning(f"Failed to parse Anthropic response content: {e}")
        return content

    def _update_metrics(self, response: ModelResponse, latency: float) -> None:
        """Update metrics with response data."""
        response_id = response.get('id', 'unknown')
        self.metrics.add_response_latency(latency, response_id)
        
        usage = response.get('usage')
        if usage:
            self.metrics.add_token_usage(
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0),
                cache_read_tokens=0,
                cache_write_tokens=0,
                context_window=0,
                response_id=response_id,
            )
        
        # Calculate cost (rough estimate)
        cost = self._calculate_cost(response)
        self.metrics.add_cost(cost)

    def _calculate_cost(self, response: ModelResponse) -> float:
        """Calculate approximate cost for the response."""
        usage = response.get('usage')
        if not usage:
            return 0.0
        
        # Cost estimates per 1K tokens (adjust based on actual pricing)
        input_cost_per_1k = self.config.input_cost_per_token * 1000 if self.config.input_cost_per_token else 0.01
        output_cost_per_1k = self.config.output_cost_per_token * 1000 if self.config.output_cost_per_token else 0.01
        
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        cost = (
            (prompt_tokens / 1000) * input_cost_per_1k +
            (completion_tokens / 1000) * output_cost_per_1k
        )
        return cost

    def vision_is_active(self) -> bool:
        """Check if vision is active (not supported for AMD models currently)."""
        return False

    def is_caching_prompt_active(self) -> bool:
        """Check if prompt caching is active."""
        return False

    def is_function_calling_active(self) -> bool:
        """Check if function calling is active (AMD gateway uses mock function calling)."""
        return self._function_calling_active

    def get_token_count(self, messages: list[dict]) -> int:
        """Get approximate token count for messages."""
        # Simple approximation: ~4 chars per token
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        return total_chars // 4

    def __str__(self) -> str:
        return f'AmdLLM(model={self.model_name})'

    def __repr__(self) -> str:
        return str(self)


def is_amd_model(model: str) -> bool:
    """Check if the model string indicates an AMD LLM model."""
    return (
        model.startswith('amd/') or 
        'llm-api.amd.com' in model or
        model in ('gpt-5', 'gpt-5-codex', 'gpt-5.1', 'claude-opus-4.5', 'claude-sonnet-4.5')
    )
