import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from novelrag.llm.logger import LLMRequest, LLMResponse, log_llm_call
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class LLMMixin:
    """Mixin that provides LLM template calling functionality"""
    
    def __init__(self, template_env: TemplateEnvironment, chat_llm: BaseChatModel):
        self.template_env = template_env
        self.chat_llm = chat_llm

    async def call_template(self, template_name: str, user_question: str | None = None, json_format: bool = False, **kwargs: Any) -> str:
        """Call an LLM with a template and return the response."""
        logger.info(f"Calling template: {template_name} with json_format={json_format} ─────────────────")
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        logger.debug('\n' + prompt)
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        
        # Prepare request for logging
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please answer the question.'}
        ]
        request = LLMRequest(
            messages=messages,
            response_format='json_object' if json_format else None
        )
        
        # Call LLM and measure time
        start_time = time.time()
        lc_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_question or 'Please answer the question.'),
        ]
        invoke_kwargs: dict[str, Any] = {}
        if json_format and getattr(self.chat_llm, '_json_supports', True):
            invoke_kwargs["response_format"] = {"type": "json_object"}
        result: AIMessage = await self.chat_llm.ainvoke(lc_messages, **invoke_kwargs)
        response = result.content if isinstance(result.content, str) else json.dumps(result.content)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log the call
        log_response = LLMResponse(content=response)
        log_llm_call(template_name, request, log_response, duration_ms)
        
        logger.info(f"Received response from LLM for template {template_name}")
        logger.info('\n' + response) 
        logger.info('───────────────────────────────────────────────────────────────────────────────')
        return response

    async def call_template_structured(self, template_name: str, response_schema: dict[str, Any], user_question: str | None = None, **kwargs: Any) -> str:
        """Call an LLM with a template and enforce a specific JSON schema for the response.
        
        Args:
            template_name: Name of the template to load and render
            response_schema: JSON schema definition that the response must conform to
            user_question: Optional user question to include in the chat
            **kwargs: Template variables to pass to the template renderer
            
        Returns:
            The LLM response as a string (should be valid JSON conforming to the schema)
        """
        logger.info(f"Calling template: {template_name} with structured schema ─────────────────")
        template = self.template_env.load_template(template_name)
        prompt = template.render(**kwargs)
        logger.debug('\n' + prompt)
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        logger.debug(f"Response schema: {json.dumps(response_schema, indent=2)}")
        logger.debug('───────────────────────────────────────────────────────────────────────────────')
        
        # Prepare request for logging
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': user_question or 'Please provide a response that conforms to the specified JSON schema.'}
        ]
        response_format_config = {
            "type": "json_schema", 
            "json_schema": {
                "name": "structured_response",
                "schema": response_schema,
                "strict": True
            }
        }
        request = LLMRequest(
            messages=messages,
            response_format=json.dumps(response_format_config)  # Store as JSON string for logging
        )
        
        # Call LLM and measure time
        start_time = time.time()
        lc_messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_question or 'Please provide a response that conforms to the specified JSON schema.'),
        ]
        invoke_kwargs: dict[str, Any] = {}
        if getattr(self.chat_llm, '_json_supports', True):
            invoke_kwargs["response_format"] = response_format_config
        result: AIMessage = await self.chat_llm.ainvoke(lc_messages, **invoke_kwargs)
        response = result.content if isinstance(result.content, str) else json.dumps(result.content)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log the call
        log_response = LLMResponse(content=response)
        log_llm_call(template_name, request, log_response, duration_ms)
        
        logger.info(f"Received structured response from LLM for template {template_name}")
        logger.info('\n' + response) 
        logger.info('───────────────────────────────────────────────────────────────────────────────')
        return response
