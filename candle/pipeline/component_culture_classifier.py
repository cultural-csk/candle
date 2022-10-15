import logging
from functools import partial
from typing import List

import pymongo
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

from ds.sentence_culture_labels_item import SentenceCultureLabelsItem
from ds.sentence_item import SentenceItem
from pipeline.pipeline_component import PipelineComponent
from utils.mongodb_handler import get_database

try:
    set_start_method("spawn")
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


class TextSet(Dataset):
    def __init__(self, texts: List[str]):
        self._texts = texts

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, index):
        return self._texts[index]


def single_run(sentence_items: List[SentenceItem], device: int,
               model_name: str, batch_size: int, labels: List[str],
               db_host: str, db_port: int, db_name: str, collection_name: str):
    # Load the classifier if it's not loaded
    logger.info(
        f"Loading the classifier \"{model_name}\" "
        f"to device \"{device}\"...")
    classifier = pipeline("zero-shot-classification",
                          model=model_name, device=device)
    # Get the texts
    texts = [item.text for item in sentence_items]

    # Classify the sentences
    logger.info(
        f"Classifying {len(texts):,} sentences "
        f"using the classifier \"{model_name}\" "
        f"on {(len(texts) // batch_size + 1):,} batches...")
    sentence_culture_labels_item = []
    text_set = TextSet(texts)
    results = []
    for result in tqdm(classifier(text_set, labels,
                                  batch_size=batch_size,
                                  multi_label=True),
                       total=len(texts)):
        results.append(result)

    # Get the labels
    for sentence_item, result in zip(sentence_items, results):
        scores = {
            label: score for label, score in
            zip(result["labels"], result["scores"])
        }
        sentence_culture_labels_item.append(
            SentenceCultureLabelsItem(
                None, sentence_item, scores))
    logger.info(
        f"Classified {len(sentence_culture_labels_item):,} sentences.")

    # Insert the culture labels into the database
    logger.info(
        f"Inserting {len(sentence_culture_labels_item):,} "
        f"culture labels into the database...")
    client = pymongo.MongoClient(db_host, db_port)
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(
        [item.to_dict() for item in sentence_culture_labels_item])
    logger.info(
        f"Inserted {len(sentence_culture_labels_item):,} culture labels "
        f"into the database.")


class CultureClassifier(PipelineComponent):
    description = "Classify sentences into elements of culture"
    config_layer = ["pipeline_components", "culture_classifier"]

    def __init__(self, config: dict):
        super().__init__(config)

        # Get local config
        self._local_config = config
        for layer in self.config_layer:
            self._local_config = self._local_config[layer]

        # Get the labels
        self._candidate_labels = self._local_config["candidate_labels"]
        self._counter_labels = self._local_config.get("counter_labels", [])
        self._all_labels = self._candidate_labels + self._counter_labels

        # Get the classifier config
        self._model_name = self._local_config["model"]
        self._devices = config["gpus"]
        self._classifier = None

        # Get the database config
        db_config = self._local_config["db_collections"]

        # Assign the database collections
        db = get_database(**config["mongo_db"])
        self._sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentences']['name']}"]
        self._sentence_culture_labels_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentence_culture_labels']['name']}"]
        self._generic_sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['generic_sentences']['name']}"]

        # Index the collections
        db_config = self._local_config["db_collections"]
        indexes = db_config["sentence_culture_labels"]["indexes"]
        for index in indexes:
            field = index["field"]
            unique = index.get("unique", False)
            self._sentence_culture_labels_collection.create_index(field,
                                                                  unique=unique)
        for index_name in self._candidate_labels:
            self._sentence_culture_labels_collection.create_index(
                [(f"scores.{index_name}", pymongo.DESCENDING)])
        for index_name in self._counter_labels:
            self._sentence_culture_labels_collection.create_index(
                [(f"scores.{index_name}", pymongo.ASCENDING)])

    def run(self):
        """ Classifies the sentences into elements of culture """
        # Get the sentences, filter by the given SpaCy file lists
        logger.info("Getting the sentences from DB...")
        # generic_sentence_items = list(
        #     self._generic_sentences_collection.aggregate([
        #         {"$lookup": {
        #             "from": self._sentences_collection.name,
        #             "localField": "sentence_item_id",
        #             "foreignField": "_id",
        #             "as": "sentence_item"
        #         }},
        #         {"$unwind": "$sentence_item"},
        #         {"$match": {
        #             "$and": [
        #                 {"sentence_item.file_path": {
        #                     "$in": self._config["input"][
        #                         "spacy_file_list"]}},
        #                 {"is_generic": True},
        #             ],
        #         }},
        #     ]))
        #
        # sentence_items = [
        #     SentenceItem.from_dict(item["sentence_item"]) for item in
        #     generic_sentence_items
        # ]

        sentence_items = [
            SentenceItem.from_dict(item) for item in
            self._sentences_collection.aggregate([
                {"$lookup": {
                    "from": self._generic_sentences_collection.name,
                    "localField": "_id",
                    "foreignField": "sentence_item_id",
                    "as": "generic"
                }},
                {"$unwind": "$generic"},
                {"$match":
                    {"$and": [
                        {"file_path": {
                            "$in": self._config["input"][
                                "spacy_file_list"]
                        }},
                        {"generic.is_generic": True}
                    ]}
                },
            ])
        ]

        # Old code for running all Spacy files
        # generic_ids = [
        #     item["sentence_item_id"] for item in
        #     self._generic_sentences_collection.find({})
        # ]
        # logger.info(f"Found {len(generic_ids):,} generic sentences.")
        #
        # sentence_items = []
        # for idx in tqdm(list(range(0, len(generic_ids), 1000))):
        #     sentence_items.extend([
        #         SentenceItem.from_dict(item) for item in
        #         self._sentences_collection.find({
        #             "_id": {"$in": generic_ids[idx:idx + 1000]}
        #         })
        #     ])
        # End of old code

        logger.info(
            f"Found {len(sentence_items):,} sentences "
            f"to classify into elements of culture.")

        # Find the sentences that have already been classified
        logger.info("Getting the already classified sentences from DB...")
        existing_labels_items = []
        for idx in tqdm(list(range(0, len(sentence_items), 1000))):
            existing_labels_items.extend(list(
                self._sentence_culture_labels_collection.find(
                    {"sentence_item_id": {"$in": [s.get_id() for s in
                                                  sentence_items[
                                                  idx:idx + 1000]]}}
                )))

        if not self._local_config.get("overwrite", False):
            existing_ids = set(
                [item["sentence_item_id"] for item in existing_labels_items])
            logger.info(
                f"Found {len(existing_labels_items):,} existing culture "
                f"labels")
            sentence_items = [item for item in sentence_items
                              if item.get_id() not in existing_ids]
        else:
            logger.info(f"Removing {len(existing_labels_items):,} existing "
                        f"culture labels.")
            self._sentence_culture_labels_collection.delete_many(
                {"_id": {"$in": [s["_id"] for s in
                                 existing_labels_items]}})

        if len(sentence_items) == 0:
            logger.info("No sentences to classify.")
            return

        logger.info(
            f"Running the culture classifier on "
            f"{len(sentence_items):,} sentences...")

        # Divide into chunks corresponding to the number of GPUs
        chunks = [sentence_items[i::len(self._devices)] for i in
                  range(len(self._devices))]

        func = partial(
            single_run,
            model_name=str(self._model_name),
            batch_size=int(self._local_config.get("batch_size", 32)),
            labels=[label for label in self._all_labels],
            db_host=str(self._config["mongo_db"]["host"]),
            db_port=int(self._config["mongo_db"]["port"]),
            db_name=str(self._config["mongo_db"]["database"]),
            collection_name=str(self._sentence_culture_labels_collection.name)
        )

        # Run the component (parallelized) on the GPU(s)
        if len(self._devices) == 1:
            func(chunks[0], self._devices[0])
        else:
            with Pool(len(self._devices)) as pool:
                pool.starmap(func, zip(chunks, self._devices))

    def needs_spacy_docs(self):
        return False

    def initialize(self):
        return

    def needs_people_group_tree(self) -> bool:
        return False

    def is_initialized(self) -> bool:
        return True
