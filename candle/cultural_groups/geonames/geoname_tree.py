import json
import logging
from typing import List

from spacy.tokens import Span
from tqdm import tqdm
from treelib import Node

from cultural_groups.people_group_tree import PeopleGroupTree
from .geoname_data import GeoAlternateName, GeoName

logger = logging.getLogger(__name__)


class GeoNameTree(PeopleGroupTree):
    def __init__(self, **kwargs):
        """
        Valid keyword arguments:
        - ``all_countries_filename``: The path of the GeoNames ``allCountries.txt`` file (required)
        - ``hierarchy_filename``: The path of the GeoNames ``hierarchy.txt`` file (required)
        - ``admin1_filename``: The path of the GeoNames ``admin1CodesASCII.txt`` file (optional)
        - ``admin2_filename``: The path of the GeoNames ``admin2Codes.txt`` file (optional)
        - ``alternate_names_filename``: The path of the GeoNames ``alternateNamesV2.txt`` file (optional)
        - ``raw_demonyms_filename``: The path of the ``demonyms.json`` file (optional)
        - ``processed_demonyms_filename``: The path of the ``processed_demonyms.json`` file (optional)
        """
        super().__init__(**kwargs)

    def get_all_match_spans(self, sent: Span) -> List[Span]:
        """
        Returns all entity matches of the given sentence in the tree
        :param sent: The sentence to match (Span)
        :return: A list of entity matches (Span objects)
        """
        labels = {"NORP", "GPE"}

        matches = []
        for ent in sent.ents:
            if ent.label_ in labels:
                if self.contains_alias(ent.text):
                    matches.append(ent)

        return matches

    def sort_key(self, node: Node):
        """
        Sort key for GeoName nodes.

        Keys: population (highest to lowest), node level (lowest to highest)
        :param node: A GeoName node
        :return: The sort key
        """

        assert isinstance(node.data, GeoName), "Node data is not a GeoName"
        return -node.data.population, self.level(node.identifier)

    def _parse(self, **kwargs):
        interested_feature_codes = {
            "CONT",  # continent
            "PCL", "PCLD", "PCLF", "PCLH", "PCLI", "PCLIX", "PCLS",  # country
            "ADM1", "ADM2", "ADM3",  # administrative division
            "PPLA", "PPLA2", "PPLAG", "PPLC", "PPL",  # populated place
            # "RGN", "RGNE",  # region
        }

        root_geoname_id = "6295630"  # Earth

        all_countries_filename = kwargs["all_countries_filename"]
        hierarchy_filename = kwargs["hierarchy_filename"]
        admin1_filename = kwargs.get("admin1_filename", None)
        admin2_filename = kwargs.get("admin2_filename", None)
        alternate_names_filename = kwargs.get("alternate_names_filename",
                                              None)
        raw_demonyms_filename = kwargs.get("raw_demonyms_filename", None)
        processed_demonyms_filename = kwargs.get(
            "processed_demonyms_filename",
            None)

        logger.info("Reading all countries file...")
        geo_names = []
        root_geoname = None

        with open(all_countries_filename, "r") as f:
            for line in tqdm(f):
                line = line.strip("\n")
                if not line:
                    continue
                geo_name = GeoName.from_string(line)

                if geo_name.geoname_id == root_geoname_id:
                    root_geoname = geo_name
                else:
                    geo_names.append(geo_name)

        logger.info(f"Read {len(geo_names):,} GeoNames of interest.")

        assert root_geoname is not None, "Root GeoName not found."

        logger.info("Reading hierarchy file...")
        child_id_2_parent_id = {}
        with open(hierarchy_filename, "r") as f:
            for line in tqdm(f):
                line = line.strip("\n")
                if not line:
                    continue
                parent_id, child_id, _ = line.split("\t")

                # only take the first parent occurring in the hierarchy file
                if child_id in child_id_2_parent_id:
                    continue

                child_id_2_parent_id[child_id] = parent_id

        logger.info(f"Read {len(child_id_2_parent_id):,} parent-child pairs.")

        admin_to_geoname_id = {}
        if admin1_filename:
            logger.info("Reading admin1 file...")
            with open(admin1_filename, "r") as f:
                for line in tqdm(f):
                    line = line.strip("\n")
                    if not line:
                        continue
                    code, _, _, geoname_id = line.split("\t")
                    admin_to_geoname_id[code] = geoname_id

        if admin2_filename:
            logger.info("Reading admin2 file...")
            with open(admin2_filename, "r") as f:
                for line in tqdm(f):
                    line = line.strip("\n")
                    if not line:
                        continue
                    code, _, _, geoname_id = line.split("\t")
                    admin_to_geoname_id[code] = geoname_id

        if admin_to_geoname_id:
            logger.info(
                f"Read {len(admin_to_geoname_id):,} admin-geoname_id pairs.")

            logger.info("Additional hierarchy information...")
            cnt = 0
            for geo_name in tqdm(geo_names):
                if geo_name.geoname_id in child_id_2_parent_id:
                    continue
                admin_id = f"{geo_name.country_code}.{geo_name.admin1code}"
                if geo_name.admin2code and geo_name.admin2code != "00":
                    admin_id = f"{geo_name.country_code}." \
                               f"{geo_name.admin1code}." \
                               f"{geo_name.admin2code}"

                if admin_id in admin_to_geoname_id:
                    child_id_2_parent_id[geo_name.geoname_id] = \
                        admin_to_geoname_id[admin_id]
                    cnt += 1
            logger.info(f"Added {cnt:,} additional parent-child pairs.")

        logger.info("Remove dead ends...")
        to_remove = set()
        for child_id, parent_id in child_id_2_parent_id.items():
            if child_id == parent_id:
                to_remove.add(child_id)
        for child_id in to_remove:
            del child_id_2_parent_id[child_id]
        logger.info(f"Removed {len(to_remove):,} dead ends.")

        logger.info("Creating nodes...")
        nodes = []
        for geo_name in tqdm(geo_names):
            if geo_name.feature_code == "PPL" and (int(
                    geo_name.population) < 1000):
                continue

            if geo_name.feature_code not in interested_feature_codes:
                continue

            if geo_name.feature_code in {"PPLA2", "ADM3"} \
                    and geo_name.admin2code in {"", "00"}:
                continue

            nodes.append(Node(
                tag=geo_name.name,
                identifier=geo_name.geoname_id,
                data=geo_name,
            ))

        logger.info(f"Created {len(nodes):,} nodes.")

        logger.info("Building tree...")

        logger.info("Adding nodes to tree...")
        self.create_node(tag=root_geoname.name,
                         identifier=root_geoname.geoname_id, data=root_geoname)

        for node in tqdm(nodes):
            self.add_node(node, parent=root_geoname_id)

        logger.info(f"Moving nodes to their true parents...")
        for node in tqdm(nodes):
            if node.identifier in child_id_2_parent_id:
                parent_id = child_id_2_parent_id[node.identifier]
                num_try = 0
                while parent_id != root_geoname_id:
                    if self.contains(parent_id):
                        break

                    if parent_id not in child_id_2_parent_id or num_try > 10:
                        logger.warning(
                            f"Node {node.identifier} <- {parent_id} "
                            f"not found in hierarchy file. "
                            f"Number of tries: {num_try}.")
                        break

                    parent_id = child_id_2_parent_id[parent_id]
                    num_try += 1

                if self.contains(parent_id):
                    if node.data.feature_code != "CONT" \
                            and parent_id == root_geoname_id:
                        logger.warning(
                            f"{node.identifier} ({node.data.name}) is "
                            f"not a continent. Removing from self.")
                        self.remove_node(node.identifier)
                    else:
                        if parent_id != child_id_2_parent_id[
                            node.identifier]:
                            logger.warning(
                                f"{node.identifier} ({node.data.name})"
                                f" is not a direct child of {parent_id} "
                                f"({self.get_node(parent_id).data.name}).")
                        self.move_node(node.identifier, parent_id)
                else:
                    logger.warning(
                        f"Could not find a parent for "
                        f"{node.identifier} ({node.data.name}). "
                        f"Removing from self.")
                    self.remove_node(node.identifier)
            else:
                logger.warning(
                    f"{node.identifier} ({node.data.name}) "
                    f"is not in hierarchy file. Removing from self.")
                self.remove_node(node.identifier)

        logger.info(f"Built tree with {len(self.all_nodes()):,} nodes.")

        if alternate_names_filename:
            logger.info("Reading alternate names file...")
            geo_alt_names = {}
            with open(alternate_names_filename, "r") as f:
                for line in tqdm(f):
                    line = line.strip("\n")
                    if not line:
                        continue
                    alt_name = GeoAlternateName.from_string(line)

                    if not self.contains(alt_name.geoname_id):
                        continue

                    if alt_name.geoname_id not in geo_alt_names:
                        geo_alt_names[alt_name.geoname_id] = []
                    geo_alt_names[alt_name.geoname_id].append(alt_name)
            logger.info(
                f"Read {sum(len(v) for v in geo_alt_names.values()):,} "
                f"alternate names.")

            logger.info("Updating GeoNames.Org alternate names...")
            cnt = 0
            for node in tqdm(self.all_nodes()):
                if node.identifier in geo_alt_names:
                    node.data.geo_alternate_names = geo_alt_names[
                        node.identifier]
                    node.data.update_alias_set()
                    cnt += 1
            logger.info(
                f"Updated GeoNames.Org alternate names of {cnt:,} nodes.")

        if processed_demonyms_filename:
            logger.info("Reading processed demonyms file...")
            with open(processed_demonyms_filename, "r") as f:
                geoname_id_2_demonym = json.load(f)
            logger.info(
                f"Read {len(geoname_id_2_demonym):,} processed demonyms.")

            logger.info("Updating processed demonyms...")
            cnt = 0
            for node in tqdm(self.all_nodes()):
                if node.identifier in geoname_id_2_demonym:
                    node.data.demonyms = geoname_id_2_demonym[node.identifier]
                    node.data.update_alias_set()
                    cnt += 1
            logger.info(
                f"Updated processed demonyms of {cnt:,} nodes.")
        elif raw_demonyms_filename:
            logger.info("Reading demonyms file...")
            with open(raw_demonyms_filename, "r") as f:
                demonym_2_places = json.load(f)
            place_2_demonyms = {}
            for demonym, places in demonym_2_places.items():
                for place in places:
                    if place not in place_2_demonyms:
                        place_2_demonyms[place] = []
                    place_2_demonyms[place].append(demonym)
            logger.info(
                f"Read {len(demonym_2_places):,} demonyms and "
                f"{len(place_2_demonyms):,} places.")

            logger.info("Updating demonyms...")
            cnt = 0
            for place, demonyms in tqdm(place_2_demonyms.items()):
                geo_entities = sorted(
                    self.filter_nodes(lambda x: x.data.has_alias(place)),
                    key=lambda x: (
                        -x.data.population,
                        self.level(x.identifier)
                    )
                )
                if geo_entities:
                    geo_entities[0].data.demonyms += demonyms
                    geo_entities[0].data.update_alias_set()
                    cnt += 1
            logger.info(f"Updated demonyms of {cnt:,} nodes.")

    def save_hierarchy_to_file(self, filename):
        with open(filename, "w") as f:
            for node in self.all_nodes():
                parent = self.parent(node.identifier)
                if parent is None:
                    continue

                parent_id = parent.identifier
                child_id = node.identifier
                _type = "DERV"

                f.write(f"{parent_id}\t{child_id}\t{_type}\n")

    def save_alternate_names_to_file(self, filename):
        with open(filename, "w") as f:
            for node in self.all_nodes():
                geo_entity: GeoName = node.data
                for alt_name in geo_entity.geo_alternate_names:
                    f.write(
                        f"{alt_name.alternate_name_id}\t"
                        f"{alt_name.geoname_id}\t"
                        f"{alt_name.iso_language}\t"
                        f"{alt_name.alternate_name}\t"
                        f"{1 if alt_name.is_preferred_name else ''}\t"
                        f"{1 if alt_name.is_short_name else ''}\t"
                        f"{1 if alt_name.is_colloquial else ''}\t"
                        f"{1 if alt_name.is_historic else ''}\t"
                        f"{alt_name.from_date}\t"
                        f"{alt_name.to_date}\n")

    def save_demonyms_to_file(self, filename):
        processed_demonyms = {}
        for node in self.all_nodes():
            geo_entity: GeoName = node.data
            if geo_entity.demonyms:
                processed_demonyms[geo_entity.geoname_id] = list(
                    set(geo_entity.demonyms))
        with open(filename, "w") as f:
            json.dump(processed_demonyms, f, indent=4)
