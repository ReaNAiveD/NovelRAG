import abc

from novelrag.agent.steps import StepDefinition, StepOutcome


class PlanningStrategy(abc.ABC):
    def initial_planning_instructions(self) -> list[str]:
        raise NotImplementedError()

    def adapt_planning_instructions(self) -> list[str]:
        raise NotImplementedError()

    def filter_planning_context(self, context: dict[str, list[str]]) -> dict[str, list[str]]:
        raise NotImplementedError()
    
    def post_planning(self, merged_steps: list[StepDefinition]) -> list[StepDefinition]:
        raise NotImplementedError()
    

class ContextStrategy(abc.ABC):
    def extract_context_instructions(self) -> list[str]:
        raise NotImplementedError()


class CompositePlanningStrategy(PlanningStrategy):
    def __init__(self, strategies: list[PlanningStrategy]):
        self.strategies = strategies

    def initial_planning_instructions(self) -> list[str]:
        result = []
        for i, strategy in enumerate(self.strategies):
            instructions = strategy.initial_planning_instructions()
            if instructions:
                result.extend(instructions)
                if i < len(self.strategies) - 1:  # Not the last strategy
                    result.append("")
        return result

    def adapt_planning_instructions(self) -> list[str]:
        result = []
        for i, strategy in enumerate(self.strategies):
            instructions = strategy.adapt_planning_instructions()
            if instructions:
                result.extend(instructions)
                if i < len(self.strategies) - 1:  # Not the last strategy
                    result.append("")
        return result

    def filter_planning_context(self, context: dict[str, list[str]]) -> dict[str, list[str]]:
        for strategy in self.strategies:
            context = strategy.filter_planning_context(context)
        return context
    
    def post_planning(self, merged_steps: list[StepDefinition]) -> list[StepDefinition]:
        for strategy in self.strategies:
            merged_steps = strategy.post_planning(merged_steps)
        return merged_steps


class CompositeContextStrategy(ContextStrategy):
    def __init__(self, strategies: list[ContextStrategy]):
        self.strategies = strategies
    
    def extract_context_instructions(self) -> list[str]:
        result = []
        for i, strategy in enumerate(self.strategies):
            instructions = strategy.extract_context_instructions()
            if instructions:
                result.extend(instructions)
                if i < len(self.strategies) - 1:  # Not the last strategy
                    result.append("")
        return result


class NoOpPlanningStrategy(PlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return []

    def initial_planning_instructions(self) -> list[str]:
        return self.shared_planning_instructions() + []

    def adapt_planning_instructions(self) -> list[str]:
        return self.shared_planning_instructions() + []

    def filter_planning_context(self, context: dict[str, list[str]]) -> dict[str, list[str]]:
        return context
    
    def post_planning(self, merged_steps: list[StepDefinition]) -> list[StepDefinition]:
        return merged_steps


class ResourceStructureKnowledge(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "Resources follow a hierarchical URI structure:",
            "- Root URI (`/`) returns all available aspect names",
            "- Aspect URIs (`/{aspect}`) return aspect metadata and child resource names",
            "- Resource URIs (`/{aspect}/{resource}`) return individual resources and their child names",
            "- Child resources are accessed by appending `/{child_name}` to the parent URI"
        ]


class AspectStructureKnowledge(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "When planning resource operations, consider aspect structure and dependencies:",
            "- Check relationships field for cross-aspect dependencies (e.g., character requires CNE context)",
            "- Respect constraints to ensure valid resource creation and modification (e.g., unique IDs, domain non-overlap)", 
            "- Use metadata.required_fields and optional_fields to guide data gathering steps"
        ]
    

class AspectQueryInstructions(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "When planning context expansion after aspect queries, consider these optional search directions:",
            "- Fetch related aspects via relationships field if they strongly align with current objectives",
            "- Fetch the most promising root resources that directly support the task intent",
            "- Search for resources that match aspect characteristics relevant to the goal",
            "- Stop expanding when sufficient context is gathered for effective planning",
        ]


class ResourceQueryInstructions(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "When planning context expansion after resource queries, consider these optional search directions:",
            "- Fetch parent aspect only if resource context suggests critical missing information",
            "- Search for highly relevant cross-domain resources following aspect relationships",
            "- Fetch parent/child resources when they appear central to achieving the objective",
            "- Search for related resources selectively based on content relevance and task importance",
            "- Prioritize depth over breadth - focus on the most promising discovery paths",
        ]


class ResourceCreateInstructions(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "When planning resource creation steps, follow these overarching guidelines:",
            "1. Fetch the root uri (`/`) to list all available aspects.",
            "2. Identify and Fetch the target aspect for the new resource based on task requirements.",
            "2.1. Sepecially, if there is no suitable aspect, create a new aspect first.",
            "3. Review and expand the context based on the aspect's definition, constraints, and relationships.",
            "3.1. Keep expand the aspects and resources context until you have sufficient information to create the resource.",
            "3.2. In order to avoid infinite expansion, limit the number of expansion steps to 10.",
            "4. Create the new resource under the target aspect or target parent resource. Ensure compliance with aspect constraints and relationships. Provide all related context in the creation step.",
        ]


class ResourceUpdateInstructions(NoOpPlanningStrategy):
    def shared_planning_instructions(self) -> list[str]:
        return [
            "When planning resource update steps, follow these overarching guidelines:",
            "1. Fetch the target resource to understand its current state and attributes.",
            "2. Fetch the aspect of the target resource to understand its definition, constraints, and relationships.",
            "3. Review and expand the context based on the resource and aspect information.",
            "3.1. Keep expand the aspects and resources context until you have sufficient information to update the resource.",
            "3.2. In order to avoid infinite expansion, limit the number of expansion steps to 10.",
            "4. Update the target resource with the new information. Ensure compliance with aspect constraints and relationships. Provide all related context in the update step.",
        ]


class NoOpContextStrategy(ContextStrategy):
    def extract_context_instructions(self) -> list[str]:
        return []


class ResourceContextKnowledge(NoOpContextStrategy):
    def extract_context_instructions(self) -> list[str]:
        return [
            "Resources follow a hierarchical URI structure, including root, aspect, resource, and multiple levels of child resources.",
            "If the context is extracted from a root, aspect or resource, include the URI path in the context to clarify its level in the hierarchy.",
            "The URI should be appended to the context sentence. For example, 'Annie always works hard at TechCorp(/organization/techcorp) to win awards(/achievements/techcorp_award/excellence_award). (URI: /characters/annie)'.",
            "When referencing other resources in the context, include their URIs as well.",
        ]
