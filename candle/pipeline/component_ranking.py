import csv
import json
import logging
import re
from typing import Tuple, List, Set
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from spacy.tokens import Doc, Span
from tqdm import tqdm
from unidecode import unidecode

from cultural_groups import PeopleGroup
from pipeline.pipeline_component import PipelineComponent
from utils.mongodb_handler import get_database
from utils.representative import get_first_sentence
from utils.spacy_reader import SPACY_MODEL_NAME

logger = logging.getLogger(__name__)

STOP_WORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
              "you", "your", "yours", "yourself", "yourselves", "he", "him",
              "his", "himself", "she", "her", "hers", "herself", "it", "its",
              "itself", "they", "them", "their", "theirs", "themselves", "what",
              "which", "who", "whom", "this", "that", "these", "those", "am",
              "is", "are", "was", "were", "be", "been", "being", "have", "has",
              "had", "having", "do", "does", "did", "doing", "a", "an", "the",
              "and", "but", "if", "or", "because", "as", "until", "while", "of",
              "at", "by", "for", "with", "about", "against", "between", "into",
              "through", "during", "before", "after", "above", "below", "to",
              "from", "up", "down", "in", "out", "on", "off", "over", "under",
              "again", "further", "then", "once", "here", "there", "when",
              "where", "why", "how", "all", "any", "both", "each", "few",
              "more", "most", "other", "some", "such", "no", "nor", "not",
              "only", "own", "same", "so", "than", "too", "very", "s", "t",
              "can", "will", "just", "don", "should", "now"}

BAD_CONCEPTS = {
    "food": {"menu", "word", "food", "foods", "cuisine", "cuisines", "dish",
             "dishes", "meal", "meals", "breakfast", "lunch", "dinner",
             "restaurant", "restaurants", "served", "serving", "serves",
             "hotel", "hotels", "eat", "eats", "used", "cooked"},
    "drink": {"menu", "restaurant", "restaurants", "drink", "drinks", "word",
              "food", "foods", "dish", "dishes", "meal", "meals", "breakfast",
              "lunch", "dinner"},
    "clothing": {"clothing", "clothes", "word"},
    "tradition": {"traditions", "word", "tradition", "traditional"},
    "ritual": {"ritual", "rituals", "word"},
}

COMMON_NAMES = {"Smith", "Anderson", "Clark", "Wright", "Mitchell", "Johnson",
                "Thomas", "Rodriguez", "Lopez", "Perez", "Williams", "Jackson",
                "Lewis", "Hill", "Roberts", "Jones", "White", "Lee", "Scott",
                "Turner", "Brown", "Harris", "Walker", "Green", "Phillips",
                "Davis", "Martin", "Hall", "Adams", "Campbell", "Miller",
                "Thompson", "Allen", "Baker", "Parker", "Wilson", "Garcia",
                "Young", "Gonzalez", "Evans", "Moore", "Martinez", "Hernandez",
                "Nelson", "Edwards", "Taylor", "Robinson", "King", "Carter",
                "Collins", "James", "David", "Christopher", "George", "Ronald",
                "John", "Richard", "Daniel", "Kenneth", "Anthony", "Robert",
                "Charles", "Paul", "Steven", "Kevin", "Michael", "Joseph",
                "Mark", "Edward", "Jason", "William", "Thomas", "Donald",
                "Brian", "Jeff", "Mary", "Jennifer", "Lisa", "Sandra",
                "Michelle", "Patricia", "Maria", "Nancy", "Donna", "Laura",
                "Linda", "Susan", "Karen", "Carol", "Sarah", "Barbara",
                "Margaret", "Betty", "Ruth", "Kimberly", "Elizabeth", "Dorothy",
                "Helen", "Sharon", "Deborah", "Jonathan", "George", "Stephen",
                "Julia", "Emily", "Carolyn", "Jessica", "Amanda", "Melissa",
                "Heather", "Amy", "Angela", "Michelle", "Laura", "Sarah",
                "Kimberly", "Stephanie", "Nicole", "Christine", "Rebecca",
                "Kelly", "Teresa", "Sandra", "Donna", "Patricia", "Cynthia",
                "Sharon", "Kathleen", "Deborah", "Alicia", "Denise", "Tammy",
                "Angela", "Brenda", "Melissa", "Amy", "Anna", "Debra",
                "Virginia", "Katherine", "Pamela", "Catherine", "Ruth",
                "Christina", "Samantha", "Janet", "Debbie", "Carol", "Julie",
                "Lori", "Martha", "Andrea", "Frances", "Ann", "Alice",
                "Mitch", "Juha", "Igor", "Jari", "Jukka", "Jussi", "Jyrki", }

GENERAL_PATTERNS = [
    re.compile(
        r"(^for ((example)|(instance)|(e\.g\.)))",
        re.IGNORECASE),
    re.compile(
        r"("
        r"\(\d+\)|"
        r"(sentence \d+)|"
        r"(((the)|(this)|(these)|(those)|(that)|(each)|(all))( \w+)? "
        r"sentences?)|"
        r"(in this list)"
        r")",
        re.IGNORECASE),
    re.compile(
        r"\bHD wallpaper\b",
        re.IGNORECASE),
]

FOOD_DRINK_PATTERN = re.compile(
    r"\b((the menu)|"
    r"(dining ((rooms?)|(areas?)))|"
    r"(french doors?)|"
    r"(brazil(ian)? nuts?)|"
    r"(vietnam veterans?)|"
    r"(unsung hero(es)?)|"
    r"(((the)|(this)|(these)) restaurants?)|"
    r"(North American bison)|"
    r"(will be served)|"
    r"(was served)|"
    r"(the food here)|"
    r"(will be a mix)|"
    r"(dining options))\b",
    re.IGNORECASE)

DOMAIN_PATTERN = {
    "clothing": GENERAL_PATTERNS + [
        re.compile(r"\b(german shepherds*)\b", re.IGNORECASE)],
    "food": GENERAL_PATTERNS + [FOOD_DRINK_PATTERN],
    "drink": GENERAL_PATTERNS + [FOOD_DRINK_PATTERN],
    "tradition": GENERAL_PATTERNS + [
        re.compile(r"\b(german shepherds*)\b", re.IGNORECASE)],
    "ritual": GENERAL_PATTERNS,
    "religious": GENERAL_PATTERNS,
}

PLURAL_EXCEPTIONS = {
    "fries",
}

BAD_OCCUPATIONS_WORDS = {
    "™", "lvl"
}

BAD_OCCUPATIONS_PATTERN = re.compile(
    r"\b(((nanny|childcare) agenc(y|ies))|(Four Seasons)|(Nanny McPhee)|"
    r"(net nanny)|(Boss Design)|(Iyengar Yoga)|(Two Inch Astronaut)|"
    r"(Hiawatha Care Center)|(Hilltop Manor)|(Reg Barber)|(Elm Street Pomade)|"
    r"(Crafted North)|(Jacques Bar)|(Flair Bartender)|(The Hitman's Bodyguard)|"
    r"(Results:)|(Label:)|(Butcher Box)|(Belsize Park London)|(Jeepney routes)|"
    r"(Top Cars)|(Image Luxury Cars)|(Area of specialty:)|(A K M Studio)|"
    r"(Urban Style)|(Hair Affair)|(Kudos Hair)|(Hip Headz)|"
    r"(Pure Hair And Beauty)|(Gel Triq)|(Hair & Beauty Club)|"
    r"(M & M Hairdressing)|(Chow chows)|"
    r"(is an?( \w)? ((hairdressers?)|(sculptors?)|(artists?)))|"
    r"(Magic Oz)|(Reed Tire)|(PC tune-up)|(Drum On)|(A Choice Nanny)|"
    r"(Martindale)|(Tutunov Piano Series)|(Logbook Pro)|(the situation)|"
    r"(Douglas-Sarpy)|(Sailor Pluto)|(Dacron)|(Dubarry's)|(Salesman:)|"
    r"(Hyunn-Min)|(Casino Luck)|(NextGen Gaming)|(IGT)|"
    r"(About Company Novomatic)|(BPH)|"
    r"(Play'n Go)|(Net Entertainment)|(Realtime Gaming))\b",
    re.IGNORECASE
)

OCCUPATIONS_DOC_FILTERS = [
    # not starting with some words
    lambda doc: doc[0].text.lower() not in {"the", "both", "none", "no",
                                            "every"},

    # does not have named entities
    lambda doc: not any(ent.label_ in {"PERSON", "NORP", "ORG", "GPE", "LOC",
                                       "LAW", "LANGUAGE", "DATE", "TIME",
                                       "PERCENT", "MONEY", "QUANTITY",
                                       "ORDINAL", "CARDINAL"} for ent in
                        doc.ents),
    # does not have pronouns
    lambda doc: not any((token.pos_ == "PRON" and
                         token.text.lower() in {"he",
                                                "she",
                                                "his",
                                                "her",
                                                "him"})
                        for token in doc),

    # does not have "will"
    lambda doc: not any(
        token.text.lower() == "will" and token.pos_ == "AUX" for token in doc),

    # does not have names
    lambda doc: not any(token.text in COMMON_NAMES for token in doc),

    # does not have "this", "these", etc.
    lambda doc: not any(token.text.lower() in {"this", "these"}
                        for token in doc),

    # does not have "were", "was"
    lambda doc: not any(
        token.text.lower() in {"were", "was", "have"} and token.pos_ == "AUX"
        for token in doc),

    # does not have bad words
    lambda doc: not any(
        token.text.lower() in BAD_OCCUPATIONS_WORDS for token in doc),

    # does not have uppercase words
    lambda doc: not any(
        (token.text.isupper() and len(token.text) >= 4) for token in doc),

    # root word is not close to the end
    lambda doc: len(doc) - list(doc.sents)[0].root.i > 2,

    # does not match regex
    lambda doc: not BAD_OCCUPATIONS_PATTERN.search(doc.text),
]


def get_domain(url: str):
    """Get the domain of a URL."""
    parsed_uri = urlparse(url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    return domain


class RankingComponent(PipelineComponent):
    description = "Ranks statements"
    config_layer = ["pipeline_components", "ranking_component"]

    def __init__(self, config):
        super().__init__(config)

        # Get local config
        self._local_config = config
        for layer in self.config_layer:
            self._local_config = self._local_config[layer]

        self._local_config["aspect_map"] = self._local_config.get(
            "aspect_map", {})

        self.ids = self._local_config["input"]["ids"]
        self._target_label = self._local_config["input"]["label"]

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
        self._clusters_with_reps_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['clusters_with_reps']['name']}_{self._target_label}"]

        # Models
        self._sbert = None
        self._spacy_model = None
        self._people_group_tree = None
        self._form_of = None

    def run(self):
        if not self.is_initialized():
            self.initialize()

        logger.info(f"Running {self.__class__.__name__}")

        logger.info(f"Querying database")
        cluster_items = list(
            self._clusters_with_reps_collection.find(
                {"node_id": {"$in": self.ids}}))
        logger.info(
            f"Found {len(cluster_items):,} cluster items with representatives")

        logger.info(
            f"There are "
            f"{sum(len(c['sentence_item_ids']) for c in cluster_items):,} "
            f"sentences in the clusters")

        merge_rows = self.filter_clusters_based_on_representatives(
            cluster_items)

        logger.info("Extracting concepts")
        for row in tqdm(merge_rows):
            row["concepts"] = self.extract_concepts(
                row,
                group=self._people_group_tree.get_node(row["node_id"]).data,
            )

        merge_rows = [row for row in merge_rows if len(row["concepts"]) > 0]

        bs = [self.build_masked_sentence(s) for s in tqdm(merge_rows)]
        masked_sentences = [b[0] for b in bs]
        spacy_docs = [b[1] for b in bs]
        has_subjects = [b[2] for b in bs]
        matches_list = [b[3] for b in bs]

        logger.info("Compute embeddings")
        embeddings = self._sbert.encode(masked_sentences,
                                        convert_to_tensor=True,
                                        show_progress_bar=True)

        logger.info(f"Computing similarities")
        cosine_scores = util.cos_sim(embeddings, embeddings)

        logger.info(f"Computing scores")
        num_nouns = []
        pass_spacy_filters = []
        for doc, row in zip(spacy_docs, merge_rows):
            cluster_count = 0
            for t in doc:
                if t.pos_ == "NOUN":
                    cluster_count += 1
            num_nouns.append((cluster_count + 1) / (len(doc) + 1))

            pass_spacy_filter = True
            if self._config["people_group"] == "occupations":
                for func in OCCUPATIONS_DOC_FILTERS:
                    if not func(doc):
                        pass_spacy_filter = False
                        break

                if pass_spacy_filter:
                    node_data: PeopleGroup = self._people_group_tree.get_node(
                        row["node_id"]).data
                    aliases = node_data.get_aliases()

                    for alias in aliases:
                        if not alias:
                            continue
                        if re.search(rf"\bthe {alias}", doc.text,
                                     re.IGNORECASE):
                            pass_spacy_filter = False
                            break

            pass_spacy_filters.append(pass_spacy_filter)

        # freqs = torch.tensor([r["size"] for r in merge_rows]).int()
        # max_freqs = freqs.max(dim=0)
        subject2max_freq = {}
        for row, has_subject, pass_filter in zip(merge_rows, has_subjects,
                                                 pass_spacy_filters):
            if not (has_subject and pass_filter):
                continue

            subject = row["subject"]
            if subject not in subject2max_freq:
                subject2max_freq[subject] = 1
            subject2max_freq[subject] = max(subject2max_freq[subject],
                                            row["size"])

        idfs = torch.Tensor(
            cosine_scores >= self._local_config["idf_threshold"]).sum(dim=0)
        idfs = (idfs.shape[0] / idfs).log()
        max_idf = idfs.max()
        idfs = idfs / max_idf

        for row, idf, n, has_subject, pass_filter in zip(merge_rows,
                                                         idfs.tolist(),
                                                         num_nouns,
                                                         has_subjects,
                                                         pass_spacy_filters):
            tf = min(
                row["size"] / subject2max_freq.get(row["subject"], row["size"]),
                1.0)
            row["tf"] = tf
            row["idf"] = idf
            row["nouns"] = n
            row["has_subjects"] = has_subject
            row["pass_spacy_filter"] = pass_filter

        logger.info(f"Create new rows")
        new_rows = []
        for row, doc, masked_sentence, matches in tqdm(
                list(zip(merge_rows, spacy_docs, masked_sentences,
                         matches_list))):
            sentences = [{
                "text": s["text"],
                "url": s["url"],
            } for s in row["cluster"]]

            cs = sum([row["prob"], row["tf"], row["idf"], row["nouns"]]) / 4
            if not (row["has_subjects"] and row["pass_spacy_filter"]):
                cs = 0.0

            new_row = {
                "id": str(row["id"]),
                "subject": row["subject"],
                "domain": self._config.get("domain",
                                           self._config["people_group"]),
                "rep": row["rep"],
                "size": row["size"],
                "prob": row["prob"],
                "tf": row["tf"],
                "idf": row["idf"],
                "noun_density": row["nouns"],
                "has_subjects": row["has_subjects"],
                "combined_score": cs,
                "sentences": sentences,
                "aspect": self._local_config["aspect_map"].get(
                    self._target_label, self._target_label),
                "concepts": row["concepts"],
                "masked_sentence": masked_sentence,
                "tokens": [t.text for t in doc],
                "matches": [{
                    "text": m.text,
                    "start": m.start,
                    "end": m.end,
                } for m in matches],
            }

            if new_row["concepts"] and cs > 0:
                new_rows.append(new_row)

        logger.info("Filtering concepts")
        concept2subjects = {}
        for new_row in new_rows:
            subject = new_row["subject"]
            for concept in new_row["concepts"]:
                if concept not in concept2subjects:
                    concept2subjects[concept] = set()
                concept2subjects[concept].add(subject)

        bad_concepts = BAD_CONCEPTS.get(self._target_label,
                                        {self._target_label,
                                         self._target_label + "s"})
        good_concepts = set([concept for concept in concept2subjects if
                             len(concept2subjects[concept]) / len(self.ids) <
                             self._local_config["concept_threshold"]
                             and concept not in bad_concepts])

        logger.info(
            f"Writing result to file {self._local_config['output']['file']}")
        with open(self._local_config["output"]["file"], "w+") as func:
            cluster_count = 0
            concept_count = 0
            for new_row in new_rows:
                new_row["concepts"] = [c for c in new_row["concepts"] if
                                       c in good_concepts]
                cluster_count += 1
                concept_count += len(new_row["concepts"])
                func.write(json.dumps(new_row))
                func.write("\n")

        logger.info(f"Found {concept_count:,} concepts in {cluster_count:,} "
                    f"clusters")

    def needs_spacy_docs(self) -> bool:
        return False

    def initialize(self):
        if self.is_initialized():
            return
        logger.info(f"Initializing {self.__class__.__name__}")
        if self._sbert is None:
            self._sbert = SentenceTransformer(
                self._local_config["sbert"]["model"])
        if self._people_group_tree is None:
            self._people_group_tree = self._config["people_group_tree"]
        if self._spacy_model is None:
            logger.info(f"Loading spacy model")
            self._spacy_model = spacy.load(SPACY_MODEL_NAME)
        if self._form_of is None:
            logger.info(f"Loading ConceptNet FormOf relations")
            self._form_of = {}
            with open(self._local_config["conceptnet_form_of"], "r") as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader):
                    if row["head"] not in self._form_of:
                        self._form_of[row["head"]] = row["tail"]

    def needs_people_group_tree(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self._sbert is not None and self._people_group_tree is not None \
               and self._spacy_model is not None and self._form_of is not None

    def filter_clusters_based_on_representatives(self, cluster_items):
        rows = []
        for c in tqdm(cluster_items):
            cluster = list(
                self._sentences_collection.find(
                    {"_id": {"$in": c["sentence_item_ids"]}}))
            rep = get_first_sentence(c["rep"])
            if not rep.endswith("."):
                continue
            if len(rep) > self._local_config["rep_filter"]["max_length"]:
                continue
            if len(rep) < 20:
                continue

            distinct_sents = sorted(set(s["text"].lower() for s in cluster),
                                    key=lambda x: len(x), reverse=True)
            sent_count = len(distinct_sents)

            if sent_count == 1:
                continue

            if sent_count / len(cluster) < 1 / 3:
                continue

            distinct_domains = set(get_domain(s["url"]) for s in cluster)
            if len(distinct_domains) / len(cluster) < 1 / 3:
                continue

            tokens = rep.split()
            if len(tokens) < 3:
                continue

            if tokens[0].lower() in {"this", "that", "these", "those",
                                     "their", "his", "her", "its"}:
                continue

            # domain-specific filters
            match = False
            for domain_pattern in DOMAIN_PATTERN.get(self._target_label,
                                                     GENERAL_PATTERNS):
                if domain_pattern.search(rep):
                    match = True
                    break
            if match:
                continue

            labels = list(self._sentence_culture_labels_collection.find(
                {"sentence_item_id": {"$in": c["sentence_item_ids"]}}))
            prob = np.mean(
                [label_item["scores"][self._target_label] for label_item in
                 labels])

            food_prob = np.mean(
                [label_item["scores"].get("food", 0) for label_item in
                 labels])
            drink_prob = np.mean(
                [label_item["scores"].get("drink", 0) for label_item in
                 labels])
            clothing_prob = np.mean(
                [label_item["scores"].get("clothing", 0) for label_item in
                 labels])
            tradition_prob = np.mean(
                [label_item["scores"].get("tradition", 0) for label_item in
                 labels])
            ritual_prob = np.mean(
                [label_item["scores"].get("ritual", 0) for label_item in
                 labels])
            behaviour_prob = np.mean(
                [label_item["scores"].get("behaviour", 0) for label_item in
                 labels])

            if self._target_label in {"tradition", "ritual"}:
                if food_prob > 0.5 or drink_prob > 0.5 or clothing_prob > 0.5:
                    continue
            if self._target_label in {"professional", "religious"}:
                if food_prob > 0.5 or \
                        drink_prob > 0.5 or \
                        clothing_prob > 0.5 or \
                        tradition_prob > 0.5 or \
                        ritual_prob > 0.5 or \
                        behaviour_prob > 0.5:
                    continue

            rows.append({
                "id": c["_id"],
                "node_id": c["node_id"],
                "rep": rep,
                "size": len(c["sentence_item_ids"]),
                "prob": prob,
                "cluster": cluster
            })
        df = pd.DataFrame(rows)

        node_rows = []
        for i in set(df["node_id"]):
            node_rows.append({
                "node_id": i,
                "subject": self._people_group_tree.get_node(
                    i).data.get_short_name(),
            })
        node_df = pd.DataFrame(node_rows)

        merge = pd.merge(node_df, df, on="node_id")

        merge_rows = merge.to_dict("records")

        return merge_rows

    def build_masked_sentence(self, row, mask_token: str = "[MASK]") \
            -> Tuple[str, Doc, int, List[Span]]:
        """Replace all matched tokens with [MASK] token."""

        rep = row["rep"]
        doc = self._spacy_model(rep)
        matches = []
        for sent in doc.sents:
            matches.extend(self._people_group_tree.get_all_match_spans(sent))
        matches = sorted(matches, key=lambda match: match.start)

        segments = []
        start_char = 0
        for m in matches:
            segments.append(rep[start_char:m.start_char])
            segments.append(mask_token)
            start_char = m.end_char
        if start_char < len(rep):
            segments.append(rep[start_char:])

        return "".join(segments), doc, int(len(matches) > 0), matches

    def get_n_gram(self, tokens, n, group: PeopleGroup = None) -> Set[str]:
        if len(tokens) < n:
            return set()

        sequences = [tokens[i:] for i in range(n)]
        res = set()
        for toks in zip(*sequences):
            if not all(t.isalpha() for t in toks):
                continue
            if any(t.lower() in STOP_WORDS for t in toks):
                continue

            gram = " ".join(t.lower() for t in toks)
            gram = unidecode(gram)
            if self._form_of is not None and gram not in PLURAL_EXCEPTIONS:
                if gram.endswith("s"):
                    gram = self._form_of.get(gram, gram)

            if group is not None:
                if any(group.has_alias(t) for t in toks):
                    continue
                if group.has_alias(gram):
                    continue

                is_part_of_name = False
                pat = re.compile(rf"\b{gram}\b", re.IGNORECASE)
                for alias in group.get_aliases():
                    if pat.search(alias):
                        is_part_of_name = True
                        break
                if is_part_of_name:
                    continue

                # for alias in group.get_aliases():
                #     pat = re.compile(f"\\b{alias}\\b")
                #     if pat.search(gram):
                #         is_part_of_name = True
                #         break
                # if is_part_of_name:
                #     continue

            res.add(gram)
        return res

    def extract_concepts(self, row, max_n=3, threshold=0.6,
                         group: PeopleGroup = None) -> List[str]:
        cluster_sents = row["cluster"]

        ngram_count = {}
        for n in range(1, max_n + 1):
            for cs in cluster_sents:
                ngrams = self.get_n_gram(cs["tokens"], n=n, group=group)
                for ng in ngrams:
                    if ng not in ngram_count:
                        ngram_count[ng] = 0
                    ngram_count[ng] += 1

        res = []
        for gram, c in sorted(ngram_count.items(), key=lambda x: x[1],
                              reverse=True):
            if c / len(cluster_sents) >= threshold:
                res.append(gram)

        to_remove = set()
        for short in res:
            toks_short = short.split()
            for long in res:
                toks_long = long.split()
                if len(toks_short) >= len(toks_long):
                    continue
                pat = re.compile(f"\\b{short}\\b")
                if pat.search(long):
                    to_remove.add(short)
                    break

        return [r for r in res if r not in to_remove]
