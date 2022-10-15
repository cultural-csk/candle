from typing import List

from cultural_groups import PeopleGroup


class SimplePeopleGroup(PeopleGroup):
    def __init__(self, name: str, aliases: List[str]):
        self._name = name.strip()
        self._aliases = list(a.strip() for a in aliases if a.strip())
        self._alias_set = set(a.lower() for a in self._aliases)

    def get_name(self) -> str:
        return self._name

    def get_short_name(self) -> str:
        return self.get_name()

    def get_id(self) -> str:
        return self.get_name()

    def get_aliases(self) -> List[str]:
        return list(self._aliases)

    def has_alias(self, alias: str) -> bool:
        if not alias.strip():
            return False
        return alias.lower() in self._alias_set
