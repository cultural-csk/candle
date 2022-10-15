import logging

from spacy.tokens import Span
from tqdm import tqdm

from ds.generic_sentence_item import GenericSentenceItem
from ds.sentence_item import SentenceItem
from pipeline.pipeline_component import PipelineComponent
from utils.mongodb_handler import get_database
from utils.spacy_reader import get_spacy_sentence, get_first_word, get_doc_url

logger = logging.getLogger(__name__)


class GenericSentenceFilter(PipelineComponent):
    description = "Filter to get generic sentences"
    config_layer = ["pipeline_components", "generic_sentence_filter"]

    def __init__(self, config: dict):
        super().__init__(config)

        # Get local config
        self._local_config = config
        for layer in self.config_layer:
            self._local_config = self._local_config[layer]

        # Get the database config
        db_config = self._local_config["db_collections"]

        # Assign the database collections
        db = get_database(**config["mongo_db"])
        self._sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentences']['name']}"]
        self._generic_sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['generic_sentences']['name']}"]

        # Index the collection
        for index in db_config["generic_sentences"]["indexes"]:
            field = index["field"]
            unique = index.get("unique", False)
            self._generic_sentences_collection.create_index(field,
                                                            unique=unique)

        # Get filter config
        self._filter_config = self._local_config["filter"]
        # Set of filters to apply to the sentences, each is a
        # function that takes a
        # SpaCy sentence (i.e., a Span) and returns a boolean.
        self._filter = {
            "is-short-enough": (
                lambda s: len(s.text.strip()) <=
                          self._filter_config["is-short-enough"]["max_length"]),
            "has-at-least-one-token": (lambda s: len(s) >= 1),
            "first-word-is-not-none": (
                lambda s: get_first_word(s) is not None and get_first_word(
                    s).text.strip() != ""),
            "starts-with-capital": (
                lambda s: get_first_word(s).text[0].isupper()),
            "ends-with-period": (lambda s: s.text.strip()[-1] == "."),
            "has-no-bad-first-word": (
                lambda s: get_first_word(s).lower_ not in
                          self._filter_config["has-no-bad-first-word"][
                              "words"]),
            "first-word-is-not-verb": (
                lambda s: get_first_word(s).pos_ != "VERB"),
            "first-word-is-not-conjunction": (
                lambda s: get_first_word(s).pos_ not in {"CCONJ", "SCONJ"}),
            "noun-exists-before-root": (lambda s: any(
                t.pos_ in {"NOUN", "PROPN"} for t in s if t.i < s.root.i)),
            "has-no-digits": (lambda s: not any(t.isdigit() for t in s.text)),

            "all-propn-have-acceptable-ne-labels": (lambda s: all(
                e.label_ not in set(self._filter_config[
                    "all-propn-have-acceptable-ne-labels"].get(
                    "excluded", [])) for e in s.ents)),

            "has-no-pronouns": (lambda s: not any((t.lower_ in set(
                self._filter_config["has-no-pronouns"][
                    "words"])) and t.pos_ == "PRON" for t in s)),

            "root-has-nsubj-or-nsubjpass": (
                lambda s: any(
                    t.dep_ in {"nsubj", "nsubjpass"} for t in s.root.children)),

            "has-no-email": (lambda s: not any(t.like_email for t in s)),
            "scr.dot_dot_in_sentence": (lambda s: ".." not in s.text),
            "scr.www_in_sentence": (lambda s: "www" not in s.text),
            "scr.com_in_sentence": (lambda s: "com" not in s.text),
            "scr.http_in_sentence": (lambda s: "http" not in s.text),

            "scr.many_hyphens_in_sentence": (
                lambda s: s.text.count("-") < 2 and s.text.count("–") < 2),
            # these are two different hyphens

            "remove-non-verb-roots": (lambda s: s.root.pos_ in {"VERB", "AUX"}),
            "remove-first-word-roots": (
                lambda s: s.root.i != get_first_word(s).i),

            "remove-present-participle-roots": (lambda s: s.root.tag_ != "VBG"),
            "remove-past-tense-roots": (lambda s: s.root.tag_ != "VBD"),

            "not-from-unreliable-source": (
                lambda s: not any(f".{d}/" in get_doc_url(s) for d in
                                  self._filter_config[
                                      "not-from-unreliable-source"][
                                      "domain_tails"])),
        }

        # Remove filters that are listed as excluded in the config
        for filter_name in self._local_config.get("excluded_filters", []):
            logger.info(f"Excluding filter {filter_name}")
            self._filter.pop(filter_name, None)

    def _is_generic(self, s: Span) -> bool:
        return not any((not func(s)) for func in self._filter.values())

    def initialize(self):
        return

    def is_initialized(self) -> bool:
        return True

    def run(self):

        logger.info("Running generic sentence filter")

        # Get the sentences
        sentence_items = list(
            SentenceItem.from_dict(d) for d in self._sentences_collection.find({
                "file_path": {"$in": self._config["input"][
                    "spacy_file_list"]}
            }))
        logger.info(f"Found {len(sentence_items):,} sentences")

        existing_generic_sentence_items = list(
            self._generic_sentences_collection.find({
                "sentence_item_id": {
                    "$in": [s.get_id() for s in sentence_items]}
            }))
        if not self._local_config.get("overwrite", False):
            existing_sentence_ids = set(
                s["sentence_item_id"] for s in existing_generic_sentence_items)
            logger.info(f"Found {len(existing_sentence_ids)} "
                        f"existing sentences")
        else:
            existing_sentence_ids = set()
            logger.info(
                f"Removing existing {len(existing_generic_sentence_items)} "
                f"generic sentence items")
            self._generic_sentences_collection.delete_many({
                "_id": {"$in": [s["_id"] for s in
                                existing_generic_sentence_items]}
            })

        sentence_items = [s for s in sentence_items if s.get_id() not in
                          existing_sentence_ids]
        logger.info(f"Filtering {len(sentence_items):,} sentences")
        spacy_sentences = [
            get_spacy_sentence(item.to_dict(), self._config["spacy_docs"])
            for item in sentence_items]

        # Filter the sentences
        is_generic_res = [self._is_generic(s) for s in tqdm(spacy_sentences)]

        logger.info(f"There are {sum(is_generic_res):,} generic sentence items")

        if sum(is_generic_res) == 0:
            logger.info("No generic sentences found")
            return

        # Write the generic sentences to the database
        generic_sentence_items = [
            GenericSentenceItem(
                _id=None,
                sentence_item=item,
                is_generic=is_generic_r)
            for item, is_generic_r in
            zip(sentence_items, is_generic_res) if is_generic_r
        ]
        logger.info(
            f"Writing {len(generic_sentence_items):,} generic sentence items "
            f"to the database")
        self._generic_sentences_collection.insert_many(
            [item.to_dict() for item in generic_sentence_items])
        logger.info("Done writing generic sentences to the database.")

    def needs_spacy_docs(self) -> bool:
        return True

    def needs_people_group_tree(self) -> bool:
        return False
