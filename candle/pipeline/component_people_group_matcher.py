import logging

import ftfy
from tqdm import tqdm

from ds.match_item import MatchItem
from ds.sentence_item import SentenceItem
from utils.mongodb_handler import get_database
from .pipeline_component import PipelineComponent

logger = logging.getLogger(__name__)


class PeopleGroupMatcher(PipelineComponent):
    description = "Finds sentences that contain people groups"
    config_layer = ["pipeline_components", "people_group_matcher"]

    def __init__(self, config: dict):
        super().__init__(config)

        self._people_group_tree = None
        self._sentences_collection = None
        self._matches_collection = None

    def initialize(self):
        # Assign the tree
        if self._people_group_tree is None:
            self._people_group_tree = self._config["people_group_tree"]

        # Get the local config
        if self._sentences_collection is None or self._matches_collection is None:
            local_config = self._config
            for layer in self.config_layer:
                local_config = local_config[layer]

            # Get the database config
            db_config = local_config["db_collections"]

            # Assign the database collections
            db = get_database(**self._config["mongo_db"])
            self._sentences_collection = \
                db[f"{self._db_collection_prefix}_" \
                   f"{db_config['sentences']['name']}"]
            self._matches_collection = db[
                f"{self._db_collection_prefix}_" \
                f"{db_config['matches']['name']}"]

            # Index the collections
            for index_name in db_config["sentences"]["indexes"]:
                self._sentences_collection.create_index(index_name)
            for index_name in db_config["matches"]["indexes"]:
                self._matches_collection.create_index(index_name)

    def run(self):
        """ Find matched entities in the SpaCy sentences """

        if not self.is_initialized():
            self.initialize()

        logger.info(
            f"Running {self.__class__.__name__} "
            f"on {len(self._config['spacy_docs'])} SpaCy files...")

        sentence_items = []
        match_items = []
        for file_path, doc_list in tqdm(self._config["spacy_docs"].items()):
            for doc_i, doc in enumerate(doc_list):
                for sent_i, sent in enumerate(doc.sents):
                    matched_nodes = set()
                    sentence_item = SentenceItem(
                        _id=None,
                        file_path=file_path,
                        doc_i=doc_i,
                        sent_i=sent_i,
                        text=ftfy.fix_text(sent.text).strip(),
                        tokens=[t.text for t in sent],
                        url=doc.user_data.get("url", None),
                    )
                    matches = self._people_group_tree.get_all_match_spans(sent)
                    if matches:
                        sentence_items.append(sentence_item)
                        for match in matches:
                            nid = self._people_group_tree.get_best_match_node(
                                match).identifier
                            if nid in matched_nodes:
                                continue
                            matched_nodes.add(nid)
                            match_item = MatchItem(
                                _id=None,
                                sentence_item=sentence_item,
                                match_node_id=nid,
                                match_text=match.text.strip(),
                                match_start=match.start - sent.start,
                                match_end=match.end - sent.start
                            )
                            match_items.append(match_item)

        if len(sentence_items) == 0:
            logger.info("No sentences found.")
            return

        logger.info(f"Inserting {len(sentence_items):,} sentence items...")
        existing_items = []
        for item in sentence_items:
            existing_items.append(self._sentences_collection.find_one(
                {"file_path": item.file_path,
                 "doc_i": item.doc_i,
                 "sent_i": item.sent_i}))

        cnt_inserted = 0
        for item, existing_item in zip(sentence_items, existing_items):
            if existing_item is None:
                res = self._sentences_collection.insert_one(item.to_dict())
                item.set_id(res.inserted_id)
                cnt_inserted += 1
            else:
                item.set_id(existing_item["_id"])

        logger.info(
            f"Inserted {cnt_inserted:,} sentence items; "
            f"the rest ({len(sentence_items) - cnt_inserted:,}) "
            f"already existed.")

        # sentence_items_res = self._sentences_collection.insert_many(
        #     [sentence_item.to_dict() for sentence_item in sentence_items])
        #
        # logger.info(f"Inserted {len(sentence_items):,} sentence items.")
        #
        # for item, _id in zip(sentence_items, sentence_items_res.inserted_ids):
        #     item.set_id(_id)

        logger.info(f"Inserting {len(match_items):,} match items...")
        self._matches_collection.insert_many(
            [m.to_dict() for m in match_items])
        logger.info(f"Inserted {len(match_items):,} match items into MongoDB.")

    def needs_spacy_docs(self) -> bool:
        return True

    def needs_people_group_tree(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return (self._people_group_tree is not None and
                self._sentences_collection is not None and
                self._matches_collection is not None)
