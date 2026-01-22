"""LLM Request/Response Logger for Agent System.

This module provides a thread-safe, memory-based logging system for LLM interactions
during agent goal pursuit. Logs are grouped by pursuit and stored in YAML format.
"""

import os
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from threading import Lock


@dataclass
class LLMRequest:
    """Represents an LLM request with all parameters."""
    messages: List[Dict[str, str]]
    response_format: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    
    def flatten(self) -> Dict[str, Any]:
        """Flatten the request into a dictionary for YAML storage."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class LLMResponse:
    """Represents an LLM response."""
    content: str
    finish_reason: Optional[str] = None
    usage_prompt_tokens: Optional[int] = None
    usage_completion_tokens: Optional[int] = None
    usage_total_tokens: Optional[int] = None
    
    def flatten(self) -> Dict[str, Any]:
        """Flatten the response into a dictionary for YAML storage."""
        result = {"content": self.content}
        for key, value in asdict(self).items():
            if key != "content" and value is not None:
                result[key] = value
        return result


@dataclass
class LLMCall:
    """Represents a single LLM call with request, response, and metadata."""
    template_name: str
    request: LLMRequest
    response: LLMResponse
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[int] = None


@dataclass
class PursuitLog:
    """Represents a goal pursuit with all its LLM calls."""
    goal: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    llm_calls: List[LLMCall] = field(default_factory=list)


class LLMLogger:
    """Thread-safe LLM logger that stores logs in memory and persists to YAML files."""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        self._current_pursuit: Optional[PursuitLog] = None
        self._all_pursuits: List[PursuitLog] = []
        self._lock = Lock()
        
        # Ensure log directory exists
        os.makedirs(log_directory, exist_ok=True)
    
    def start_pursuit(self, goal: str) -> None:
        """Start a new goal pursuit logging session."""
        with self._lock:
            # Complete previous pursuit if any
            if self._current_pursuit:
                self._current_pursuit.completed_at = datetime.now()
            
            # Start new pursuit
            self._current_pursuit = PursuitLog(goal=goal)
            self._all_pursuits.append(self._current_pursuit)
    
    def log_llm_call(
        self,
        template_name: str,
        request: LLMRequest,
        response: LLMResponse,
        duration_ms: Optional[int] = None
    ) -> None:
        """Log an LLM call to the current pursuit."""
        with self._lock:
            if not self._current_pursuit:
                raise RuntimeError("No active pursuit. Call start_pursuit() first.")
            
            call = LLMCall(
                template_name=template_name,
                request=request,
                response=response,
                duration_ms=duration_ms
            )
            self._current_pursuit.llm_calls.append(call)
    
    def complete_pursuit(self) -> None:
        """Mark the current pursuit as completed."""
        with self._lock:
            if self._current_pursuit:
                self._current_pursuit.completed_at = datetime.now()
                self._current_pursuit = None
    
    def dump_to_file(self, filename: Optional[str] = None) -> str:
        """Dump all logged pursuits to a YAML file.
        
        Args:
            filename: Optional custom filename. If not provided, uses current datetime.
            
        Returns:
            The path to the created file.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_log_{timestamp}.yaml"
        
        filepath = os.path.join(self.log_directory, filename)
        
        with self._lock:
            # Complete current pursuit before dumping
            if self._current_pursuit:
                self._current_pursuit.completed_at = datetime.now()
            
            # Convert to serializable format
            data = {
                "pursuits": [
                    {
                        "goal": pursuit.goal,
                        "started_at": pursuit.started_at.isoformat(),
                        "completed_at": pursuit.completed_at.isoformat() if pursuit.completed_at else None,
                        "llm_calls": [
                            {
                                "template_name": call.template_name,
                                "timestamp": call.timestamp.isoformat(),
                                "duration_ms": call.duration_ms,
                                "request": call.request.flatten(),
                                "response": call.response.flatten()
                            }
                            for call in pursuit.llm_calls
                        ]
                    }
                    for pursuit in self._all_pursuits
                ]
            }
        
        # Write to YAML file
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        return filepath
    
    def clear_logs(self) -> None:
        """Clear all logged pursuits from memory."""
        with self._lock:
            self._current_pursuit = None
            self._all_pursuits.clear()
    
    def get_current_pursuit_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the current pursuit."""
        with self._lock:
            if not self._current_pursuit:
                return None
            
            return {
                "goal": self._current_pursuit.goal,
                "started_at": self._current_pursuit.started_at.isoformat(),
                "llm_calls_count": len(self._current_pursuit.llm_calls),
                "templates_used": list(set(call.template_name for call in self._current_pursuit.llm_calls))
            }


# Global logger instance - will be initialized by the agent
_global_logger: Optional[LLMLogger] = None


def get_logger() -> Optional[LLMLogger]:
    """Get the global LLM logger instance."""
    return _global_logger


def initialize_llm_logger(log_directory: str = "logs") -> LLMLogger:
    """Initialize the global LLM logger."""
    global _global_logger
    _global_logger = LLMLogger(log_directory)
    return _global_logger


def log_llm_call(
    template_name: str,
    request: LLMRequest,
    response: LLMResponse,
    duration_ms: Optional[int] = None
) -> None:
    """Log an LLM call using the global logger."""
    logger = get_logger()
    if logger:
        logger.log_llm_call(template_name, request, response, duration_ms)
