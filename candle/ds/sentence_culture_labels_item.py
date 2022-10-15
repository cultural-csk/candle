from typing import Dict

import pymongo.collection

from .mongodb_item import Item
from .sentence_item import SentenceItem


class SentenceCultureLabelsItem(Item):
    def __init__(self, _id, sentence_item: SentenceItem,
                 scores: Dict[str, float]):
        self._id = _id
        self.sentence_item = sentence_item
        self.scores = scores

    def __str__(self):
        return f"{self.sentence_item} - " \
               f"{self.scores}"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "sentence_item_id": self.sentence_item.get_id(),
            "scores": self.scores
        }

    def get_id(self):
        return self._id

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        assert "sentences_collection" in kwargs, \
            "sentences_collection is required"

        sentences_collection: pymongo.collection.Collection = kwargs[
            "sentences_collection"]

        sentence_item = SentenceItem.from_dict(sentences_collection.find_one(
            {"_id": d["sentence_item_id"]}))

        return cls(d["_id"], sentence_item, d["scores"])
