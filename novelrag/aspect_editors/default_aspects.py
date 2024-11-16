from .premise import build_context as build_premise_context
from .typing import AspectContextDefinitions

default_aspects: AspectContextDefinitions = {
    'premise': build_premise_context,
}
