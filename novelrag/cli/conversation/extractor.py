from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from novelrag.utils.language import interaction_directive
from novelrag.tracer import trace_llm

SYSTEM_PROMPT = """
{language_instruction}

---Role: Task & Context Extractor---
You are a focused analyzer that processes a new task and its historical conversation to extract critical information. Your goal is to identify the latest user requirements, constraints, and relevant assistant-generated content that directly informs the new task. Prioritize accuracy by dynamically resolving conflicts, overriding outdated context, and retaining only validated or essential information.

---Guidelines---
1. Input Analysis:
    - Review the new task and historical messages in reverse chronological order.
    - Prioritize the **latest user instructions**, even if they override earlier ones.
    - Extract **user requirements** (e.g., "Use bullet points," "Avoid technical jargon") and **constraints** (e.g., "Keep under 500 words," "Exclude Section 3").
2. Assistant-Generated Content:
    - Identify outputs from the assistant that the user explicitly or implicitly approved (e.g., "Good," "Keep this section," or no objections after submission).
    - Retain these outputs **only if they are directly relevant to the new task**.
3. Conflict Resolution:
    - Treat later user inputs as overriding earlier ones.
    - If the new task partially or fully ignores historical context, discard unrelated content.
    - Merge overlapping requirements (e.g., "Use markdown" + "Add tables" → "Use markdown tables").
4. Output Logic:
    - User requirements take precedence but must coexist with validated assistant-generated content.
    - Never invent requirements or assume unstated preferences.

---Limitations---
- No Assumptions: Never invent values or infer implicit preferences.
- Do NOT retain deprecated instructions or unrelated assistant outputs.
- Ignore casual dialogue (e.g., greetings, feedback without actionable context).
- Do NOT return structured formats (e.g., JSON, lists). Use clear, concise natural language.
- Maximum length: {limit} words

---Output Format---
```
**Extracted Task Context**:  
- User Requirements: [Latest explicit/implicit demands, e.g., "Include 2023 data," "Rewrite in formal tone"]  
- Critical Assistant Content: [Relevant, user-approved outputs, e.g., "Use the framework from Slide 4"]
```

---Input---
New Task: {task}

Conversation History:
{conversation}
"""


class Extractor:
    def __init__(self, chat_llm: BaseChatModel):
        self.chat_llm = chat_llm

    @trace_llm("conversation_extract")
    async def extract(self, task: str, messages: list[str], *, limit=800) -> str:
        conversation_text = "\n".join(messages)

        # Prepare the system prompt
        system_prompt = SYSTEM_PROMPT.format(
            language_instruction=interaction_directive(language=None, is_autonomous=False),
            limit=limit,
            task=task,
            conversation=conversation_text,
        )

        result = await self.chat_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Please identify the latest user requirements, constraints, and relevant assistant-generated content according to the guidelines provided."),
        ])
        assert isinstance(result.content, str)
        return result.content.strip()
