class PeopleGroup:
    def get_name(self) -> str:
        raise NotImplementedError()

    def get_short_name(self) -> str:
        raise NotImplementedError()

    def get_id(self) -> str:
        raise NotImplementedError()

    def get_aliases(self) -> list:
        raise NotImplementedError()

    def has_alias(self, alias: str) -> bool:
        raise NotImplementedError()

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return f"PeopleGroup({self.get_id()}, {self.get_name()})"

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __hash__(self):
        return self.get_id().__hash__()
