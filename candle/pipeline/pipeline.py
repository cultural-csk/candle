import logging

from tqdm import tqdm

from cultural_groups import GeoNameTree, ReligionTree
from cultural_groups.occupations.occupation_tree import OccupationTree
from cultural_groups.regions.region_tree import RegionTree
from utils.spacy_reader import SpacyReader
from .component_clustering import ClusteringComponent
from .component_culture_classifier import CultureClassifier
from .component_generic_sentence_filter import GenericSentenceFilter
from .component_people_group_matcher import PeopleGroupMatcher
from .component_ranking import RankingComponent
from .component_rep_generator import RepresentativeGenerator
from .pipeline_component import PipelineComponent

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: dict):
        self._config = config
        self._config["db_collection_prefix"] = self.get_db_collection_prefix()

        running_component_indexes = list(
            range(min(self._config["chosen_components"]),
                  max(self._config["chosen_components"]) + 1))

        running_components = [
            self._get_possible_component_at(index) for index in
            running_component_indexes
        ]
        self._running_components = running_components

        self.print_possible_components()
        print("Run the following component:")
        print(self.get_running_components())

        if any([component.needs_spacy_docs() for component in
                self._running_components]):
            logger.info("SpaCy docs needed, loading them...")
            self._config = self._update_config_with_spacy_docs()

        if any([component.needs_people_group_tree() for component in
                self._running_components]):
            logger.info("PeopleGroupTree needed, loading it...")
            self._config["people_group_tree"] = self._init_people_group_tree()

    def run(self):
        logger.info(
            f"Running pipeline {self.__class__.__name__} with "
            f"{len(self._running_components)} "
            f"components...")
        for component in self._running_components:
            if not component.is_initialized():
                component.initialize()
            component.run()
        logger.info(f"Pipeline {self.__class__.__name__} finished.")

    def _update_config_with_spacy_docs(self):
        """ Updates the config with the actual SpaCy docs """

        spacy_reader = SpacyReader()

        file_list = self._config["input"]["spacy_file_list"]

        logger.info(f"Reading {len(file_list)} SpaCy files...")

        spacy_docs = {}
        for file_path in tqdm(file_list):
            spacy_docs[file_path] = spacy_reader.read_spacy_file(file_path)

        logger.info(
            f"Read {len(spacy_docs)} SpaCy files, which contain "
            f"{sum(len(doc_list) for doc_list in spacy_docs.values()):,} "
            f"documents.")

        self._config["spacy_docs"] = spacy_docs

        return self._config

    def get_running_components(self):
        return self._running_components

    def _get_possible_component_at(self, index) -> PipelineComponent:
        try:
            return self.get_possible_components()[index - 1](self._config)
        except IndexError:
            raise ValueError(
                f"The chosen component index {index} is out of range. "
                f"Possible components: " + str(
                    [((i + 1), c.__name__) for i, c in
                     enumerate(self.get_possible_components())]))

    @classmethod
    def print_possible_components(cls):
        print(f"Possible pipeline components of {cls.__name__}:")
        for i, component in enumerate(cls.get_possible_components()):
            print(f"  ({i + 1}) {component.__name__} - {component.description}")

    def _init_people_group_tree(self):
        raise NotImplementedError

    @classmethod
    def get_possible_components(cls):
        raise NotImplementedError

    def get_db_collection_prefix(self):
        raise NotImplementedError

    def get_cmd_arg(self):
        return self.get_db_collection_prefix()


class GeoNamePipeline(Pipeline):
    def __init__(self, config: dict):
        logger.info("Initializing GeoNamePipeline...")
        super().__init__(config)

    def _init_people_group_tree(self):
        logger.info("Initializing GeoNameTree...")
        return GeoNameTree(**self._config["geonames"])

    @classmethod
    def get_possible_components(cls):
        return [
            PeopleGroupMatcher,
            GenericSentenceFilter,
            CultureClassifier,
            ClusteringComponent,
            RepresentativeGenerator,
            RankingComponent,
        ]

    def get_db_collection_prefix(self):
        return "geonames"


class ReligionPipeline(Pipeline):
    def __init__(self, config: dict):
        logger.info("Initializing ReligionPipeline...")
        super().__init__(config)

    def _init_people_group_tree(self):
        logger.info("Initializing ReligionTree...")
        return ReligionTree(**self._config["religions"])

    @classmethod
    def get_possible_components(cls):
        return [
            PeopleGroupMatcher,
            GenericSentenceFilter,
            CultureClassifier,
            ClusteringComponent,
            RepresentativeGenerator,
            RankingComponent,
        ]

    def get_db_collection_prefix(self):
        return "religions"


class OccupationPipeline(Pipeline):
    def __init__(self, config: dict):
        logger.info("Initializing OccupationPipeline...")
        super().__init__(config)

    def _init_people_group_tree(self):
        logger.info("Initializing OccupationTree...")
        return OccupationTree(**self._config["occupations"])

    @classmethod
    def get_possible_components(cls):
        return [
            PeopleGroupMatcher,
            GenericSentenceFilter,
            CultureClassifier,
            ClusteringComponent,
            RepresentativeGenerator,
            RankingComponent,
        ]

    def get_db_collection_prefix(self):
        return "occupations"


class RegionPipeline(Pipeline):
    def __init__(self, config: dict):
        logger.info("Initializing RegionPipeline...")
        super().__init__(config)

    def _init_people_group_tree(self):
        logger.info("Initializing RegionTree...")
        return RegionTree(**self._config["regions"])

    @classmethod
    def get_possible_components(cls):
        return [
            PeopleGroupMatcher,
            GenericSentenceFilter,
            CultureClassifier,
            ClusteringComponent,
            RepresentativeGenerator,
            RankingComponent,
        ]

    def get_db_collection_prefix(self):
        return "regions"
