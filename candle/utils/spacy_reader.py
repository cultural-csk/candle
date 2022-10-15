import logging
from pathlib import Path
from typing import Union, List

import spacy
from spacy.tokens import DocBin, Doc, Span, Token

logger = logging.getLogger(__name__)

SPACY_MODEL_NAME = "en_core_web_md"


class SpacyReader:
    def __init__(self):
        self.nlp = None

    def read_spacy_file(self, filename: Union[str, Path]) -> List[Doc]:
        """
        Reads one SpaCy file and returns a list of Doc objects
        :param filename: Path to the file to read (string or Path)
        :return: List of Doc objects
        """

        if self.nlp is None:
            logger.info(f"Loading SpaCy model {SPACY_MODEL_NAME}...")
            self.nlp = spacy.load(SPACY_MODEL_NAME)
            logger.info(f"Loaded SpaCy model {SPACY_MODEL_NAME}.")

        doc_bin = DocBin().from_disk(filename)
        doc_list = list(doc_bin.get_docs(self.nlp.vocab))

        return doc_list


def get_spacy_sentence(mongo_sentence_item: dict, spacy_docs: dict) -> Span:
    """
    Takes an item from MongoDB and returns its corresponding Span object
    """
    file_path = mongo_sentence_item["file_path"]
    doc_i = mongo_sentence_item["doc_i"]
    sent_i = mongo_sentence_item["sent_i"]

    doc = spacy_docs[file_path][doc_i]
    sent = list(doc.sents)[sent_i]

    return sent


def get_first_word(sentence: Span) -> Union[None, Token]:
    """
    Returns the first word of a sentence, ignoring the heading spaces
    """
    if len(sentence) == 0:
        return None

    first_word = sentence[0]
    i = 0
    while first_word.is_space:
        if i >= len(sentence):
            return None
        first_word = sentence[i + 1]
        i += 1
    return first_word


def get_doc_url(sentence: Span) -> str:
    """
    Returns the URL of the document containing the sentence, if it exists,
    otherwise returns an empty string
    """
    return sentence.doc.user_data.get("url", "")
