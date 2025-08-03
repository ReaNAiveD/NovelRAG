from novelrag.exceptions import NoAspectSelectedError
from novelrag.resource import DirectiveElement, DirectiveElementList
from .scope import IntentScope, IntentScopeFactory


class Context:
    def __init__(self, *, aspect_factory: IntentScopeFactory):
        self.aspect_factory = aspect_factory
        self.cur_element: DirectiveElement | None = None
        self.cur_path = None
        self.current_scope: IntentScope | None = None

    async def switch(self, aspect: str | None):
        if aspect is None:
            self.current_scope = None
        else:
            self.current_scope = await self.aspect_factory.get(aspect)
        self.cur_element = None  # Reset current element when switching aspects
        self.cur_path = None

    async def cd(self, path: str):
        new_element = await self.calc_relative(path)
        if isinstance(new_element, IntentScope):
            self.cur_element = None
        elif isinstance(new_element, DirectiveElement):
            self.cur_element = new_element
        else:
            raise Exception('Unexpected Type')

    async def calc_relative(self, path: str) -> DirectiveElement | IntentScope | None:
        if not self.current_scope:
            raise NoAspectSelectedError()
        if not self.current_scope.data:
            return None
        parts = path.split('/')
        cur_element = self.cur_element
        if parts and parts[0] == '':
            # Handle absolute path
            cur_element = None
            parts = parts[1:]
        if parts and parts[-1] == '':
            parts = parts[:-1]
        parent_item: DirectiveElement | IntentScope = cur_element or self.current_scope
        cur_item: DirectiveElement | DirectiveElementList = cur_element or self.current_scope.data.root_elements
        for part in parts:
            if isinstance(cur_item, DirectiveElement):
                if len(cur_item.inner.children_keys) == 1 and part.isdecimal():
                    index = int(part)
                    key = cur_item.inner.children_keys[0]
                    parent_item = cur_item
                    cur_item = cur_item.children_of(key)[index]
                else:
                    parent_item = cur_item
                    cur_item = cur_item.children_of(part)
            elif isinstance(cur_item, DirectiveElementList):
                index = int(part)
                cur_item = cur_item[index]
            else:
                raise Exception('Unexpected Type')
        if isinstance(cur_item, DirectiveElementList):
            return parent_item
        return cur_item
