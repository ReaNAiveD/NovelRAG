from novelrag.intent import IntentContext, Action, LLMIntent
from novelrag.intent.action import Redirect
from novelrag.utils.const import LANGUAGE_INSTRUCTION

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You're an author designing Core Narrative Elements (CNEs) - self-contained story components that explore definitive aspects of a collaborative novel through specific characters, events, or world mechanics.

---Context Anchors---
Incorporate these critical inputs:
{session_summary}
{related_info}

---Objective---
1. Each CNE should Explore a NEW narrative dimension from:
    * Main plot progression/conflict escalation
    * Character-specific arc catalysts
    * Cultural/societal mechanisms
    * Environmental storytelling
    * Consequence chains
    * Thematic/emotional manifestations
    * Historical echoes
2. Avoids redundancy with existing CNEs:
{CNEs}
3. Each CNE should Contain 2+ connection hooks to existing CNEs

---Boundary Protocol---
Use these tags to define scope limits:
◼ PRESERVE: [Core element from original CNE]
◼ BRIDGE: [Underdeveloped story layer]

---Refinement Guidelines---
▸ Start with [CNE Type] header preservation check
▸ Amplify stakes while maintaining original intent
▸ Enhance 1+ cross-references using {{CNE Keyword}}
▸ Add 1 new convergence point with existing CNEs
▸ Strengthen emotional architecture through:
    * Tone consistency/amplification
    * Character motivation clarity

---Quality Validation---
CNE must pass:
☑︎ Focus Test - Maintains original aspect specialization
☑︎ Evolution Test - Shows measurable improvement
☑︎ Integration Test - Deepens ecosystem connections
☑︎ Consistency Test - Aligns with established canon

---CNE Example Structure---
[Environmental Storytelling CNE]
The sentient stormfront "Maelis" migrates westward, its lightning encoding warnings about {{Location CNE}}. As it approaches {{Character CNE}}'s homeland:
* AVOID: Repeating storm mechanics from {{Weather System CNE}}
* BRIDGE: Unexplained visions in {{Prophecy CNE}}
    1. Disrupts {{Technology Subplot CNE}} through EM pulses
    2. Reveals corruption patterns matching {{Artifact CNE}}
    3. Mirrors {{Ancient Migration CNE}} routes in animal behavior
       Emotional Core: Activates ancestral dread in Character A while fueling Character B's redemption quest

---Output Structure Specifications---

* Main body in plain text format
* When modifying CNE list, insert a single code block:
```modification
[JSON array of operation commands]
```

---Operation Command JSON Schema---

* Mandatory "type" field for all objects
    * type=element:
        * Requires start/end/data fields
        * start: insertion index (0-based)
        * end: deletion end index (exclusive)
        * data: array of JSON objects to insert
    * type=property:
        * Requires element_id/data fields
        * element_id: target element UUID
        * data: key-value pairs for update
Example

```modification
[
  {{
    "type": "element",
    "start": 1,
    "end": 3,
    "data": [
      {{"cne": "FAKE CNE 1"}},
      {{"cne": "FAKE CNE 2"}}
    ]
  }},
  {{
    "type": "property",
    "element_id": "FAKE_UUID",
    "data": {
      "cne": "FAKE CNE 3"
    }}
  }}
]
```
This example demonstrates:
* At index 1: Delete original elements 1-2, insert two new elements
* For element 4: Update priority/timeout fields while preserving others
* Single code block compliance maintained
"""

class Default(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return '_default'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        if not message:
            return Action(
                redirect=Redirect(
                    aspect='cne',
                    intent='list',
                    message=None,
                )
            )

        history = await context.conversation_history.get_messages(aspect='cne', intent=None)
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(message)

        system_prompt = SYSTEM_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTION,
            session_summary=summary,
            related_info=related,
            CNEs=[ele for ele in context.current_aspect.root_elements]
        )
        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Create a new CNE according to the guidelines.'}
        ])

