from typing import Callable, Dict, Any

from novelrag.aspect import AspectContext

AspectContextDefinitions = dict[str, Callable[[Dict[str, Any]], AspectContext]]
