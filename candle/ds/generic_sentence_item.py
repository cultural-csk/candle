import pymongo.collection

from .mongodb_item import Item
from .sentence_item import SentenceItem


class GenericSentenceItem(Item):
    def __init__(self, _id, sentence_item: SentenceItem,
                 is_generic: bool):
        self._id = _id
        self.sentence_item = sentence_item
        self.is_generic = is_generic

    def to_dict(self):
        return {
            "sentence_item_id": self.sentence_item.get_id(),
            "is_generic": self.is_generic
        }

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        assert "sentences_collection" in kwargs, \
            "sentences_collection is required"

        sentences_collection: pymongo.collection.Collection = kwargs[
            "sentences_collection"]

        sentence_item = SentenceItem.from_dict(sentences_collection.find_one(
            {"_id": d["sentence_item_id"]}))

        return cls(d["_id"], sentence_item, d["is_generic"])
