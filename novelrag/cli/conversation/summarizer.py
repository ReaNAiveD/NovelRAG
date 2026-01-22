from novelrag.llm import ChatLLM
from novelrag.utils.const import LANGUAGE_INSTRUCTION

SYSTEM_PROMPT = """
{language_instruction}

---Role: Dynamic Conversation Summarizer---

You are a temporal-aware summarization engine that condenses chat histories while preserving evolving context. Prioritize recency-weighted compression without losing critical decision pathways.

---Guidelines---

1. Chronological Processing:
    - Process messages in reverse chronological order to identify decision anchors
    - Maintain original timeline when writing the summary
2. Detail Allocation:
    - Allocate 50% of summary space to the most recent 30% of conversation
    - For recurring topics: track first occurrence, major changes, and final consensus
3. Information Triage:
    - Mandatory retention:
        - Explicit user directives ("Never use...")
        - System-critical data (numbers, names, deadlines)
        - Resolved conflicts/contradictions
    - Optional retention:
        - Repeated phrases indicating emphasis
        - Successful clarification exchanges
4. Update Mechanism:
    - When updating existing summaries:
        1. Preserve verified factual backbone
        2. Overlay new developments as foreground
        3. Prune obsolete context older than 3 message turns

---Output Format---

**Conversation Digest**  
[Time-sensitive summary with clear temporal markers (e.g., "Initially... Later... Finally"). Recent developments contain 2-3x more specific details than older content. Strictly adhere to {limit} words.]

---Input Conversation---

Existing Summary: {existing_summary}

{conversation}
"""


class Summarizer:
    def __init__(self, chat_llm: ChatLLM):
        self.chat_llm = chat_llm

    async def summarize(self, messages: list[str], *, existing_summary: str | None = None, limit=800) -> str:
        conversation_text = "\n".join(messages)

        # Prepare the system prompt
        system_prompt = SYSTEM_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTION,
            limit=limit,
            existing_summary=existing_summary or "No Existing Summary",
            conversation=conversation_text
        )
        
        # Get response from LLM using proper message format
        response = await self.chat_llm.chat([
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "Please summarize the conversation according to the guidelines provided."
            }
        ])

        return response.strip()
