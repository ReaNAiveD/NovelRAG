from novelrag.llm import ChatLLM
from .extractor import Extractor
from .item import ConversationItem
from .summarizer import Summarizer


class ConversationHistory:
    def __init__(
        self, 
        conversations: list[ConversationItem], 
        *,
        summarizer: Summarizer | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        """
        Initialize conversation history.
        
        Args:
            conversations: List of conversation items to initialize with
        """
            
        self.conversations = conversations
        self.summarization: str | None = None
        self.summarization_end_index: int = 0
        self.summarizer = summarizer
        self.extractor = extractor

    @classmethod
    def empty(cls, *, chat_llm: ChatLLM | None = None) -> "ConversationHistory":
        """
        Create an empty conversation history.
        """
        return cls(
            conversations=[],
            summarizer=Summarizer(chat_llm) if chat_llm else None,
            extractor=Extractor(chat_llm) if chat_llm else None,
        )

    async def get_messages(
            self,
            aspect: str | None,
            intent: str | None,
            *,
            max_messages: int = 4
    ) -> list[dict[str, str]]:
        """
        Retrieve consecutive contextual messages for request construction.

        Messages are collected in reverse chronological order until context breaks,
        then returned in chronological order. Context matching rules:

        - When aspect=None: Use latest message's aspect, collect consecutive messages
          with the same aspect (intent ignored)
        - When aspect specified:
            - With intent: Collect consecutive messages with matching aspect AND intent
            - Without intent: Collect consecutive messages with matching aspect only

        Args:
            aspect: Target context aspect (None to inherit from latest message)
            intent: Optional sub-context intent (requires explicit aspect)
            max_messages: Maximum messages to return (default: 4, must be > 0)

        Returns:
            Chronologically ordered messages with 'role' and 'content' keys,
            containing up to max_messages most recent consecutive matches

        Raises:
            ValueError: Invalid max_messages or intent without explicit aspect
        """
        # Parameter validation
        if max_messages <= 0:
            raise ValueError("max_messages must be a positive integer")

        if not self.conversations:
            return []

        # Resolve target context
        target_aspect = aspect or self.conversations[-1].aspect
        check_intent = aspect is not None and intent is not None

        collected = []
        for item in reversed(self.conversations):
            if item.aspect != target_aspect:
                break

            if check_intent and item.intent != intent:
                break

            message_parts = []
            if item.aspect:
                message_parts.append(f"[aspect: {item.aspect}]")
            if item.intent:
                message_parts.append(f"[intent: {item.intent}]")
            if item.message:
                message_parts.append(f": {item.message}")
            collected.insert(0, {"role": item.role, "content": "".join(message_parts)})

        return collected[-max_messages:]

    async def _summary_to(
        self, 
        new_end_index: int, 
        *, 
        summarizer: Summarizer,
        reuse_old: bool = True, 
        summary_limit: int = 800
    ) -> str:
        """
        Summarize conversation up to the specified index.
        
        Args:
            new_end_index: Index up to which to summarize
            summarizer: Summarizer instance to use
            reuse_old: Whether to reuse previous summary
            summary_limit: Maximum word limit for the summary
        
        Returns:
            str: Generated summary of the conversation
            
        Raises:
            ValueError: If new_end_index exceeds conversation length
        """
        if new_end_index > len(self.conversations):
            raise ValueError("new_end_index cannot exceed conversation length")
            
        if new_end_index <= 0:
            return ""

        # Collect messages to summarize
        messages: list[str] = []
        
        # Add messages from last summary point to new end index
        start_idx = self.summarization_end_index if reuse_old else 0
        for item in self.conversations[start_idx:new_end_index]:
            # Format message with role and optional metadata
            message_parts = [f"{item.role}"]
            if item.aspect:
                message_parts.append(f"[aspect: {item.aspect}]")
            if item.intent:
                message_parts.append(f"[intent: {item.intent}]")
            if item.message:
                message_parts.append(f": {item.message}")
            messages.append("".join(message_parts))
        
        # Don't summarize if no messages to process
        if not messages:
            return ""

        # Generate new summary
        return await summarizer.summarize(
            messages=messages,
            existing_summary=self.summarization if reuse_old else None,
            limit=summary_limit,
        )

    async def get_summary(self, *, reuse_old = True, limit = 800):
        end_index = len(self.conversations)
        summary = await self._summary_to(
            new_end_index=end_index,
            summarizer=self.summarizer,
            reuse_old=reuse_old,
            summary_limit=limit)
        self.summarization_end_index = end_index
        self.summarization = summary
        return summary

    async def extract_related(
            self,
            task: str,
            *,
            max_messages = 10,
            limit = 800
    ):
        messages = []
        prev_aspect = None
        for item in list(reversed(self.conversations))[:max_messages]:
            if prev_aspect is None and item.aspect:
                prev_aspect = item.aspect
            elif prev_aspect is not None and item.aspect is not None and item.aspect != prev_aspect:
                break
            message_parts = [f"{item.role}"]
            if item.aspect:
                message_parts.append(f"[aspect: {item.aspect}]")
            if item.intent:
                message_parts.append(f"[intent: {item.intent}]")
            if item.message:
                message_parts.append(f": {item.message}")
            messages.append("".join(message_parts))
        messages = messages[::-1]
        return await self.extractor.extract(task, messages, limit=limit)

    def add_user(self, message: str, *, aspect: str | None = None, intent: str | None = None) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: The user's message
            aspect: Optional aspect classification
            intent: Optional intent classification
        """
        if not message.strip():
            return
        self.conversations.append(ConversationItem(
            role='user',
            aspect=aspect,
            intent=intent,
            message=message,
        ))

    def add_assistant(self, message: str, *, aspect: str | None = None, intent: str | None = None) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: The assistant's message
            aspect: Optional aspect classification
            intent: Optional intent classification
        """
        if not message.strip():
            return
        self.conversations.append(ConversationItem(
            role='assistant',
            aspect=aspect,
            intent=intent,
            message=message,
        ))
