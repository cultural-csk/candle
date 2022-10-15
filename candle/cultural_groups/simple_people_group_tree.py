import csv
import logging
from typing import List

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from treelib import Node

from cultural_groups import PeopleGroupTree, SimplePeopleGroup
from utils.spacy_reader import SPACY_MODEL_NAME

logger = logging.getLogger(__name__)


class SimpleTree(PeopleGroupTree):
    data_model = SimplePeopleGroup

    def __init__(self, **kwargs):
        """
        Valid keyword arguments:
        - ``filepath``: The path to the data file in tsv format (required)
        """
        super().__init__(**kwargs)

        logger.info("Load Spacy model...")
        self._spacy_model = spacy.load(SPACY_MODEL_NAME)

        logger.info("Create Matcher...")
        self._matcher = PhraseMatcher(self._spacy_model.vocab, attr="LOWER")
        patterns = [self._spacy_model.make_doc(alias) for node in
                    self.all_nodes() for alias in node.data.get_aliases()]
        self._matcher.add("ALIASES", patterns)

    def _parse(self, **kwargs) -> None:
        assert "filepath" in kwargs, "religions_filepath is a " \
                                     "required argument"

        filepath = kwargs["filepath"]

        logger.info(f"Parsing {filepath}...")
        rows = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = row["name"].strip()
                aliases = [a.strip() for a in row["aliases"].strip().split(",")
                           if a.strip()]
                rows.append({
                    "name": name,
                    "aliases": aliases
                })
        logger.info(f"{len(rows)} rows parsed.")

        logger.info("Creating tree...")
        religions = [self.data_model(**row) for row in rows]
        # create root node
        root_id = "ROOT"
        self.create_node(tag=root_id,
                         identifier=root_id,
                         data=self.data_model(root_id, []))
        # nodes as a flat list
        for religion in religions:
            self.create_node(tag=religion.get_name(),
                             identifier=religion.get_id(),
                             data=religion,
                             parent=root_id)

    def sort_key(self, node: Node):
        return node.data.get_name()

    def get_all_match_spans(self, sent: Span) -> List[Span]:
        """
        Returns all spans that match any of the aliases in the tree
        :param sent: The sentence to search in
        :return: A list of spans that match any of the aliases in the tree
        """
        return self._matcher(sent, as_spans=True)
