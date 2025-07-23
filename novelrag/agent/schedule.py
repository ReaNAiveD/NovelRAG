"""Scheduling system for agent execution."""

import json
import uuid
from datetime import datetime
from typing import Any

from .types import StepStatus


class Step:
    """Represents a single execution step in the agent schedule."""
    
    def __init__(self, tool: str, description: str, step_id: str | None = None, 
                 request_from: 'Step | None' = None, chain_from: 'Step | None' = None, 
                 rescheduled_from: 'Step | None' = None):
        self.step_id = step_id or uuid.uuid4().hex
        self.tool = tool
        self.description = description
        self.status = StepStatus.PENDING
        # selected content proposal
        self.proposal: str | None = None
        self.result: list[Any] = []
        # The step from which this step was created through Step Decomposition
        self.request_from = request_from
        # The step from which this step was created through Chained Update
        self.chain_from = chain_from
        # The step from which this step was created through rescheduling
        self.rescheduled_from = rescheduled_from
        self.additional_context: dict[str, Any] = {}
        self.created_at = datetime.now()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error_message: str | None = None

    @classmethod
    def create_simple(cls, tool: str, description: str) -> 'Step':
        """Create a simple step with minimal configuration"""
        return cls(tool=tool, description=description)
    
    @classmethod
    def create_decomposed(cls, tool: str, description: str, parent_step: 'Step') -> 'Step':
        """Create a step from decomposition"""
        return cls(tool=tool, description=description, request_from=parent_step)
    
    @classmethod
    def create_chained(cls, tool: str, description: str, trigger_step: 'Step') -> 'Step':
        """Create a step from chain update"""
        return cls(tool=tool, description=description, chain_from=trigger_step)
    
    def is_completed(self) -> bool:
        """Check if step is completed"""
        return self.status == StepStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if step failed"""
        return self.status == StepStatus.FAILED
    
    def is_executing(self) -> bool:
        """Check if step is currently executing"""
        return self.status == StepStatus.EXECUTING
    
    def get_duration(self) -> float | None:
        """Get step execution duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class Scheduler:
    """Manages the execution schedule and handles dynamic schedule modifications"""
    
    def __init__(self, initial_steps: list[Step]):
        self.pending_steps: list[Step] = initial_steps.copy()
        self.executing_step: Step | None = None
        self.completed_steps: list[Step] = []
        self.failed_steps: list[Step] = []
        
    def get_current_step(self) -> Step | None:
        return self.executing_step
        
    def start_next_step(self) -> Step | None:
        if not self.pending_steps:
            return None
        step = self.pending_steps.pop(0)
        step.status = StepStatus.EXECUTING
        step.started_at = datetime.now()
        self.executing_step = step
        return step
        
    def complete_current_step(self):
        if self.executing_step:
            self.executing_step.status = StepStatus.COMPLETED
            self.executing_step.completed_at = datetime.now()
            self.completed_steps.append(self.executing_step)
            self.executing_step = None
            
    def fail_current_step(self, error_message: str):
        if self.executing_step:
            self.executing_step.status = StepStatus.FAILED
            self.executing_step.error_message = error_message
            self.executing_step.completed_at = datetime.now()
            self.failed_steps.append(self.executing_step)
            self.executing_step = None
            
    def add_decomposed_steps(self, parent_step: Step, decomposed_steps: list[dict[str, str]], insert_at_front: bool = True):
        """Add steps from decomposition"""
        new_steps = []
        for step_data in decomposed_steps:
            new_step = Step.create_decomposed(
                tool=step_data['tool'],
                description=step_data['description'],
                parent_step=parent_step
            )
            new_steps.append(new_step)
        
        if insert_at_front:
            self.pending_steps = new_steps + self.pending_steps
        else:
            self.pending_steps.extend(new_steps)
            
    def add_chain_update_steps(self, triggering_step: Step, chain_updates: list[str]):
        """Add steps from chain updates"""
        # Implementation would parse chain_updates and create new steps
        # with chain_from=triggering_step
        # For now, we'll add them as simple steps
        new_steps = []
        for update in chain_updates:
            # Parse update string to extract tool and description
            # For simplicity, assume format "tool:description"
            if ":" in update:
                tool, description = update.split(":", 1)
                new_step = Step.create_chained(
                    tool=tool.strip(),
                    description=description.strip(),
                    trigger_step=triggering_step
                )
                new_steps.append(new_step)
        self.pending_steps.extend(new_steps)
        
    def add_backlog_items(self, items: list[str]):
        """Add items to backlog (could be converted to steps later)"""
        # For now, we'll treat backlog items as low-priority steps
        # In a full implementation, this would interact with a separate backlog system
        pass
    
    def get_pending_steps(self) -> list[Step]:
        """Get a copy of pending steps"""
        return self.pending_steps.copy()
    
    def get_completed_steps(self) -> list[Step]:
        """Get a copy of completed steps"""
        return self.completed_steps.copy()
    
    def get_failed_steps(self) -> list[Step]:
        """Get a copy of failed steps"""
        return self.failed_steps.copy()
    
    def clear_failed_steps(self):
        """Clear the failed steps list"""
        self.failed_steps.clear()
    
    def has_pending_steps(self) -> bool:
        """Check if there are pending steps"""
        return len(self.pending_steps) > 0
        
    def build_context_for_step(self, target_step: Step) -> list[str]:
        """Build context by traversing step relationships"""
        need_context = [target_step]
        result = []
        
        # Traverse completed steps in reverse chronological order
        for step in self.completed_steps[::-1]:
            if self._is_step_relevant(step, need_context):
                need_context.append(step)
                result.append(json.dumps({
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "description": step.description,
                    "status": step.status,
                    **step.additional_context,
                    "result": step.result,
                }))
        return result
        
    def _is_step_relevant(self, step: Step, context_steps: list[Step]) -> bool:
        """Check if a step is relevant to the context"""
        return (step.request_from in context_steps or 
                step.chain_from in context_steps or 
                step.rescheduled_from in context_steps)



