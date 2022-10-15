import logging
import sys
from typing import List, Union

from spacy.tokens import Span
from treelib import Tree, Node

from .people_group import PeopleGroup

logger = logging.getLogger(__name__)


class PeopleGroupTree(Tree):
    def __init__(self, **kwargs):
        super().__init__()

        self._parse(**kwargs)

        self._all_alias_list = []
        for node in self.all_nodes():
            assert isinstance(node.data,
                              PeopleGroup), "Node data is not a PeopleGroup"
            self._all_alias_list.extend(
                [alias.lower().strip() for alias in node.data.get_aliases() if
                 alias.strip()])

        self._all_alias_set = set(self._all_alias_list)

        # Cache for ``PeopleGroupTree.get_match_node()``
        logger.info(f"Creating alias_2_nodes cache...")
        self._alias_2_nodes = {}
        for node in self.all_nodes():
            for alias in node.data.get_aliases():
                alias = alias.lower().strip()
                if alias not in self._all_alias_set:
                    continue
                if alias not in self._alias_2_nodes:
                    self._alias_2_nodes[alias] = []
                self._alias_2_nodes[alias].append(node)

        for alias, nodes in self._alias_2_nodes.items():
            self._alias_2_nodes[alias] = sorted(nodes, key=self.sort_key)
        logger.info(f"alias_2_nodes cache created.")

    def _parse(self, **kwargs) -> None:
        raise NotImplementedError()

    def alias_set_size(self) -> int:
        """ Returns the number of unique aliases in the tree. """
        return len(self._all_alias_set)

    def alias_list_size(self) -> int:
        """ Returns the number of aliases in the tree. """
        return len(self._all_alias_list)

    def contains_alias(self, alias: str) -> bool:
        """ Returns True if the given alias is in the tree. """
        if not alias.strip():
            return False
        return alias.lower().strip() in self._all_alias_set

    def sort_key(self, node: Node):
        """
        Returns the sort key for the given node, used to disambiguate nodes.
        """
        raise NotImplementedError()

    def filter_nodes_by_alias(self, alias: str) -> List[Node]:
        """ Returns a list of nodes that have the given alias. """
        if not self.contains_alias(alias):
            return []
        return self._alias_2_nodes[alias]

    def get_all_match_spans(self, sent: Span) -> List[Span]:
        """ Returns a list of entity matches in the given sentence. """
        raise NotImplementedError

    def get_best_match_node(self, match: Union[str, Span]) -> Node:
        """ Returns the best node that matches the given match. """
        name = match if isinstance(match, str) else match.text
        name = name.lower().strip()
        if name not in self._alias_2_nodes:
            raise ValueError(f"No node found for {name}")
        return self._alias_2_nodes[name][0]

    def print_path_to_node(self, nid, indent: int = 2, fstream=None):
        """ Prints the path from root to the given node. """
        if indent <= 0:
            indent = 2

        if not self.contains(nid):
            return

        if fstream is None:
            fstream = sys.stdout

        for level, node_id in enumerate(reversed(list(self.rsearch(nid)))):
            if level > 0:
                fstream.write(" " * (level * indent))
                fstream.write("> ")
            data = self.get_node(node_id).data
            fstream.write(f"{data.get_name()} ({data.get_id()})\n")
