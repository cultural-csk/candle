from typing import List

from .mongodb_item import Item


class SentenceItem(Item):
    def __init__(self, _id, file_path: str, doc_i: int, sent_i: int,
                 text: str, tokens: List[str], url: str = None):
        self._id = _id
        self.file_path = file_path
        self.doc_i = doc_i
        self.sent_i = sent_i
        self.text = text
        self.tokens = tokens
        self.url = url

    def __str__(self):
        return f"{self.file_path}:{self.doc_i}:{self.sent_i} - {self.text}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.file_path == other.file_path and
                self.doc_i == other.doc_i and
                self.sent_i == other.sent_i)

    def __hash__(self):
        return hash((self.file_path, self.doc_i, self.sent_i))

    def __lt__(self, other):
        return (self.file_path, self.doc_i, self.sent_i) < (
            other.file_path, other.doc_i, other.sent_i)

    def __gt__(self, other):
        return (self.file_path, self.doc_i, self.sent_i) > (
            other.file_path, other.doc_i, other.sent_i)

    def __le__(self, other):
        return (self.file_path, self.doc_i, self.sent_i) <= (
            other.file_path, other.doc_i, other.sent_i)

    def __ge__(self, other):
        return (self.file_path, self.doc_i, self.sent_i) >= (
            other.file_path, other.doc_i, other.sent_i)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_id(self):
        return self._id

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "doc_i": self.doc_i,
            "sent_i": self.sent_i,
            "text": self.text,
            "tokens": self.tokens,
            "url": self.url
        }

    def set_id(self, _id):
        self._id = _id

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return cls(d["_id"], d["file_path"], d["doc_i"], d["sent_i"],
                   d["text"], d["tokens"], d.get("url", None))
