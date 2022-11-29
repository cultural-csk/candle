"""Script to run spaCy on the input corpus."""

import argparse
import json
import logging
import time

from spacy.tokens import DocBin

logging.basicConfig(level=logging.INFO,
                    format="[%(processName)s] "
                           "[%(asctime)s] [%(name)s] "
                           "[%(levelname)s] %(message)s",
                    datefmt="%d-%m %H:%M:%S")

logger = logging.getLogger(__name__)

SPACY_MODEL_NAME = "en_core_web_md"


def main():
    # Input file must be a JSONL file with one document per line.
    # Each document must be a JSON object with the following keys:
    # - "text": the text of the document (required)
    # - "timestamp": the timestamp of the document (optional)
    # - "url": the URL of the document (optional)

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Input file in JSONL format")
    parser.add_argument("-o", "--output_file", type=str, required=True,
                        help="Output file in spaCy DocBin format")
    parser.add_argument("-p", "--processors", type=int, default=8,
                        help="Number of processors for spaCy pipe processing")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="Batch size for spaCy pipe processing")

    args = parser.parse_args()

    logger.info("Reading input file...")
    documents = []
    with open(args.input_file, "r") as f:
        for line in f:
            documents.append(json.loads(line))
    logger.info(f"Read {len(documents):,} documents")

    logger.info("Initializing spaCy...")
    import spacy
    nlp = spacy.load(SPACY_MODEL_NAME)

    logger.info("Processing documents...")
    start_time = time.process_time()
    pipe = nlp.pipe([document["text"] for document in documents],
                    n_process=args.processors, batch_size=args.batch_size)
    docs = [doc for doc in pipe]
    # Adding metadata (timestamp and URL) to the spaCy objects
    for i, doc in enumerate(docs):
        for key in ["timestamp", "url"]:
            if key in documents[i]:
                doc.user_data[key] = documents[i][key]
    end_time = time.process_time()
    process_time = end_time - start_time
    logger.info(f"Total processing time: {process_time:.4f} seconds.")
    logger.info(
        f"Average processing time per document: "
        f"{(process_time / len(docs)):.4f} seconds.")

    logger.info(f"Writing DocBin object to disk...")
    doc_bin = DocBin(store_user_data=True, docs=docs)
    doc_bin.to_disk(args.output_file)
    logger.info("Done!")


if __name__ == "__main__":
    main()
