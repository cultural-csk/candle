import logging
from pprint import pprint
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from ds.sentence_item import SentenceItem
from pipeline.pipeline_component import PipelineComponent
from utils.clustering import hac_clustering
from utils.mongodb_handler import get_database

logger = logging.getLogger(__name__)


class ClusteringComponent(PipelineComponent):
    description = "Cluster statements"
    config_layer = ["pipeline_components", "clustering_component"]

    def __init__(self, config):
        super().__init__(config)

        # Get local config
        self._local_config = config
        for layer in self.config_layer:
            self._local_config = self._local_config[layer]

        self._ids = self._local_config["input"]["ids"]
        self._target_label = self._local_config["input"]["label"]

        # Get the database config
        db_config = self._local_config["db_collections"]

        # Assign the database collections
        db = get_database(**config["mongo_db"])
        self._sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentences']['name']}"]
        self._matches_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['matches']['name']}"]
        self._generic_sentences_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['generic_sentences']['name']}"]
        self._sentence_culture_labels_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['sentence_culture_labels']['name']}"]
        self._clusters_collection = db[
            f"{self._db_collection_prefix}_"
            f"{db_config['clusters']['name']}_{self._target_label}"]

        # Index the collection
        for index in db_config["clusters"]["indexes"]:
            field = index["field"]
            unique = index.get("unique", False)
            self._clusters_collection.create_index(field,
                                                   unique=unique)

        # SBert model
        self._sbert = None

    def run(self):
        if not self.is_initialized():
            self.initialize()

        logger.info(f"Running {self.__class__.__name__}")

        logger.info(f"Querying database")
        query_results = self.query_db()
        logger.info(f"Found {len(query_results):,} sentences")

        logger.info("Clustering statements")
        nid2clusters = self.cluster_statements(query_results)

        logger.info(f"Updating database")
        self.update_db(nid2clusters)

        logger.info(f"Done")

    def needs_spacy_docs(self) -> bool:
        return False

    def initialize(self):
        if self.is_initialized():
            return
        logger.info(f"Initializing {self.__class__.__name__}")
        self._sbert = SentenceTransformer(self._local_config["sbert"]["model"])

    def needs_people_group_tree(self) -> bool:
        return False

    def is_initialized(self) -> bool:
        return self._sbert is not None

    def update_db(self, nid2clusters: Dict[str, List[List[SentenceItem]]]):
        cluster_items = []
        for nid, clusters in nid2clusters.items():
            i_clusters = []
            for cluster_id, cluster in enumerate(clusters):
                i_clusters.append([s.get_id() for s in cluster])

            cluster_item = {
                "node_id": nid,
                "clusters": i_clusters,
            }

            cluster_items.append(cluster_item)
        if len(cluster_items) > 0:
            self._clusters_collection.insert_many(cluster_items)

    def cluster_statements(self, query_results: List[Dict]) -> Dict:
        ids_set = set(self._ids)
        nid2sentence_items: Dict[str, List[SentenceItem]] = {}
        max_sentences = self._local_config["query_db"]["max_sentences"]
        for r in query_results:
            for match in r["matches"]:
                nid = match["match_node_id"]

                if nid not in ids_set:
                    continue
                if nid not in nid2sentence_items:
                    nid2sentence_items[nid] = []

                if len(nid2sentence_items[nid]) >= max_sentences:
                    continue

                nid2sentence_items[nid].append(
                    SentenceItem.from_dict(r["sentence_item"]))

        threshold = self._local_config["hac"]["threshold"]
        nid2clusters = {}
        for nid, sentence_items in nid2sentence_items.items():
            logger.info(
                f"Clustering node {nid}: {len(sentence_items):,} sentences")
            embeddings = self._sbert.encode([s.text for s in sentence_items],
                                            show_progress_bar=True)
            raw_clusters = hac_clustering(sentence_items, embeddings,
                                          threshold)
            logger.info(f"Got {len(raw_clusters):,} clusters")
            nid2clusters[nid] = raw_clusters

        return nid2clusters

    def query_db(self) -> List[Dict]:
        config = self._local_config["query_db"]
        candidate_threshold = config["positive_threshold"]
        counter_threshold = config["negative_threshold"]

        counter_labels = self._config["counter_labels"]

        # Build the query
        label_score_gte = {
            f"scores.{self._target_label}": {"$gte": candidate_threshold}}

        counter_label_score_lte_s = []
        for c_label in counter_labels:
            counter_label_score_lte_s.append({
                f"scores.{c_label}": {"$lte": counter_threshold}
            })

        required_label_score_gte_s = []
        for required_label in self._config.get("required_true_labels", []):
            if required_label == self._target_label:
                continue
            required_label_score_gte_s.append({
                f"scores.{required_label}": {"$gte": candidate_threshold}
            })

        stage_lookup_sentences = {
            "$lookup": {
                "from": self._sentences_collection.name,
                "localField": "sentence_item_id",
                "foreignField": "_id",
                "as": "sentence_item",
            }
        }

        stage_lookup_matches = {
            "$lookup": {
                "from": self._matches_collection.name,
                "localField": "sentence_item_id",
                "foreignField": "sentence_item_id",
                "as": "matches",
            }
        }

        stage_lookup_generic_sentences = {
            "$lookup": {
                "from": self._generic_sentences_collection.name,
                "localField": "sentence_item_id",
                "foreignField": "sentence_item_id",
                "as": "generic",
            }
        }

        match_conditions = [label_score_gte]
        match_conditions.extend(counter_label_score_lte_s)
        match_conditions.extend(required_label_score_gte_s)
        match_conditions.append({"matches.match_node_id": {"$in": self._ids}})
        match_conditions.append({"generic.is_generic": True})

        pipeline = [
            stage_lookup_sentences,
            {"$unwind": "$sentence_item"},
            stage_lookup_matches,
            stage_lookup_generic_sentences,
            {"$unwind": "$generic"},
            {"$match": {"$and": match_conditions}},
        ]

        pprint(pipeline)

        return sorted(
            self._sentence_culture_labels_collection.aggregate(pipeline),
            key=lambda x: x["scores"][self._target_label], reverse=True)
