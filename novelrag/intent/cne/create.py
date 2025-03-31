from novelrag.intent import LLMIntent, IntentContext
from novelrag.intent.action import Action, UpdateAction
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import ElementOperation, AspectLocation
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
Create one original CNE (≤200 words) that:
1. Explores a NEW narrative dimension from:
    * Main plot progression/conflict escalation
    * Character-specific arc catalysts
    * Cultural/societal mechanisms
    * Environmental storytelling
    * Consequence chains
    * Thematic/emotional manifestations
    * Historical echoes
2. Avoids redundancy with existing CNEs:
{CNEs}
3. Contains 2+ connection hooks to existing CNEs

---Boundary Protocol---
Use these tags to define scope limits:
◼ AVOID: [Specific element from adjacent CNE]
◼ BRIDGE: [Underdeveloped story layer]

---Construction Guidelines---
▸ Start with [CNE Type] header
▸ Establish immediate stakes in opening line
▸ Embed 1 subtle cross-reference using {{CNE Keyword}}
▸ Include 3 convergence points with existing CNEs
▸ Demonstrate emotional architecture through:
    * Specific tone activation (hope/dread/etc)
    * 2+ character motivation links

---Quality Validation---
Your CNE must pass:
☑︎ Gap Test - Fills missing narrative layer
☑︎ Lens Test - Provides new POV/context
☑︎ Tension Test - Creates organic conflict
☑︎ Cohesion Test - Maintains world logic

---Example Structure---
[Environmental Storytelling CNE]
The sentient stormfront "Maelis" migrates westward, its lightning encoding warnings about {{Location CNE}}. As it approaches {{Character CNE}}'s homeland:
* AVOID: Repeating storm mechanics from {{Weather System CNE}}
* BRIDGE: Unexplained visions in {{Prophecy CNE}}
    1. Disrupts {{Technology Subplot CNE}} through EM pulses
    2. Reveals corruption patterns matching {{Artifact CNE}}
    3. Mirrors {{Ancient Migration CNE}} routes in animal behavior
       Emotional Core: Activates ancestral dread in Character A while fueling Character B's redemption quest
"""


class Create(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return 'create'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        history = await context.conversation_history.get_messages(aspect='cne', intent='create')
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(f"[Create Premise]{message}" if message else "Create Premise")
        system_prompt = SYSTEM_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTION,
            session_summary=summary,
            related_info=related,
            CNEs='\n'.join(['* ' + element['cne'] for element in context.current_aspect.root_elements]),
        )
        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Create a new CNE according to the guidelines.'}
        ])
        return Action(
            message=[resp],
            update=UpdateAction(
                item=PendingUpdateItem(
                    ops=[
                        ElementOperation.new(
                            location=AspectLocation.new('cne'),
                            data=[{
                                'cne': resp,
                            }]
                        )
                    ]
                )
            )
        )
