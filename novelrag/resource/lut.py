from .element import DirectiveElement


class ElementLookUpTable:
    def __init__(self, elements: list[DirectiveElement]):
        self.table: dict[str, DirectiveElement] = dict((ele.id, ele) for ele in elements)

    def find_by_id(self, id: str):
        return self.table.get(id)

    def pop(self, id: str):
        return self.table.pop(id)

    def __getitem__(self, key: str) -> DirectiveElement:
        return self.table[key]

    def __setitem__(self, key: str, value: DirectiveElement):
        self.table[key] = value
