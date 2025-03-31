from novelrag.intent import IntentContext, Action, LLMIntent, UpdateAction
from novelrag.exceptions import InvalidIndexError, InvalidMessageFormatError
from novelrag.pending_queue import PendingUpdateItem
from novelrag.resource.operation import PropertyOperation
from novelrag.utils.const import LANGUAGE_INSTRUCTION

SYSTEM_PROMPT = """
{language_instruction}

---Role---
You're an author refining Core Narrative Elements (CNEs) while maintaining their distinct role in the collaborative novel.

---Context Anchors---
Incorporate these critical inputs:
{session_summary}
{related_info}

---Objective---
Enhance CNE (≤200 words) by:
1. Strengthening its focus on its original narrative dimension
2. Addressing specific improvement areas based on user message:
{user_message}
3. Avoids redundancy with existing CNEs:
{other_CNEs}
3. Maintaining 2+ active connections with existing CNEs

---Target CNE---
{target_cne}

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
Your refined CNE must pass:
☑︎ Focus Test - Maintains original aspect specialization
☑︎ Evolution Test - Shows measurable improvement
☑︎ Integration Test - Deepens ecosystem connections
☑︎ Consistency Test - Aligns with established canon

---Example Structure---
[Updated Environmental Storytelling CNE]
The sentient stormfront "Maelis" accelerates westward (originally 15mph → now 45mph), its lightning now displaying {{Prophetic Symbol CNE}} patterns. Key changes:
* PRESERVE: Storm's sentient nature from original
* BRIDGE: Unexplained energy spikes in {{Technology CNE}}
    1. New EM pulse frequency disrupts {{Trade Route CNE}}
    2. Storm eye reveals {{Ancient Civilization CNE}} architecture
    3. Migration pattern now mirrors {{Character Journey CNE}}
       Emotional Core: Intensified ancestral dread through faster progression
"""


class Update(LLMIntent):
    @property
    def default_name(self) -> str | None:
        return 'update'

    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        try:
            split_msg = message.split(maxsplit=1)
            idx = int(split_msg[0])
            cnes = context.current_aspect.root_elements
            if idx >= len(cnes) or idx < 0:
                raise InvalidIndexError(idx, len(cnes), context.current_aspect.name)
            item = context.current_aspect.root_elements[idx]
            message = split_msg[1] if len(split_msg) > 1 else None
        except ValueError:
            raise InvalidMessageFormatError(
                self.name,
                context.aspect_name,
                message,
                "update INDEX [message]"
            )

        history = await context.conversation_history.get_messages(aspect='cne', intent='create')
        summary = await context.conversation_history.get_summary()
        related = await context.conversation_history.extract_related(f"[Update Premise #{idx}]{message}" if message else f"Update Premise #{idx}")

        system_prompt = SYSTEM_PROMPT.format(
            language_instruction=LANGUAGE_INSTRUCTION,
            session_summary=summary,
            related_info=related,
            user_message=message or 'Identify improvement areas first.',
            other_CNEs='\n'.join(['* ' + element['cne'] for ele_idx, element in enumerate(context.current_aspect.root_elements) if ele_idx != idx]),
            target_cne=item['cne'],
        )
        resp = await self.chat_llm(context.chat_llm_factory).chat(messages=[
            {'role': 'system', 'content': system_prompt},
            *history,
            {'role': 'user', 'content': message or 'Update the specified CNE according to the guidelines.'}
        ])
        return Action(
            message=[resp],
            update=UpdateAction(
                item=PendingUpdateItem(
                    ops=[
                        PropertyOperation.new(
                            element_id=item.id,
                            data={
                                'cne': resp,
                            }
                        )
                    ]
                )
            )
        )
