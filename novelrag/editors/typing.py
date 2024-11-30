from typing import Callable, Dict, Any

from novelrag.core.aspect import AspectContext
from novelrag.core.storage import NovelStorage

AspectContextDefinitions = dict[str, Callable[[NovelStorage, Dict[str, Any]], AspectContext]]
