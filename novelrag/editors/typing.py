from typing import Callable, Dict, Any

from novelrag.core.aspect import AspectContext

AspectContextDefinitions = dict[str, Callable[[Dict[str, Any]], AspectContext]]
