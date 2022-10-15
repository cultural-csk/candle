import pymongo.collection

from .mongodb_item import Item
from .sentence_item import SentenceItem


class MatchItem(Item):
    def __init__(self, _id, sentence_item: SentenceItem, match_node_id: str,
                 match_text: str, match_start: int, match_end: int):
        self._id = _id
        self.sentence_item = sentence_item
        self.match_node_id = match_node_id
        self.match_text = match_text
        self.match_start = match_start
        self.match_end = match_end

    def __str__(self):
        return f"{self.sentence_item} - " \
               f"{self.match_node_id} - " \
               f"{self.match_text} " \
               f"[{self.match_start}, {self.match_end}]"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "sentence_item_id": self.sentence_item.get_id(),
            "match_node_id": self.match_node_id,
            "match_text": self.match_text,
            "match_start": self.match_start,
            "match_end": self.match_end
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

        return cls(d["_id"], sentence_item, d["match_node_id"],
                   d["match_text"], d["match_start"], d["match_end"])
