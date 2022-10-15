class PipelineComponent:
    description = "The interface for a pipeline component"

    def __init__(self, config: dict):
        self._config = config
        self._db_collection_prefix = self._config["db_collection_prefix"]

    def run(self):
        """ Runs the component """
        raise NotImplementedError()

    def needs_spacy_docs(self) -> bool:
        """ Returns True if this component needs the SpaCy docs """
        raise NotImplementedError()

    def get_description(self):
        return self.description

    def __str__(self):
        return f"<{self.__class__.__name__}> - {self.description}"

    def __repr__(self):
        return self.__str__()

    def initialize(self):
        raise NotImplementedError()

    def needs_people_group_tree(self) -> bool:
        """ Returns True if this component needs the people group tree """
        raise NotImplementedError()

    def is_initialized(self) -> bool:
        """ Returns True if this component is initialized """
        raise NotImplementedError()
