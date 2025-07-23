from .element import DirectiveElement


class ElementLookUpTable:
    def __init__(self, elements: list[DirectiveElement]):
        self.table: dict[str, DirectiveElement] = dict((ele.uri, ele) for ele in elements)

    def find_by_uri(self, uri: str):
        return self.table.get(uri)

    def pop(self, uri: str):
        return self.table.pop(uri)

    def __getitem__(self, key: str) -> DirectiveElement:
        return self.table[key]

    def __setitem__(self, key: str, value: DirectiveElement):
        self.table[key] = value
