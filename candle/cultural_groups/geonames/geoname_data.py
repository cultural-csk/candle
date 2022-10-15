from typing import List

from cultural_groups import PeopleGroup


class GeoAlternateName(object):
    def __init__(self,
                 alternate_name_id: str,
                 geoname_id: str,
                 iso_language: str,
                 alternate_name: str,
                 is_preferred_name: bool,
                 is_short_name: bool,
                 is_colloquial: bool,
                 is_historic: bool,
                 from_date: str,
                 to_date: str):
        self.alternate_name_id = alternate_name_id.strip()
        self.geoname_id = geoname_id.strip()
        self.iso_language = iso_language.strip()
        self.alternate_name = alternate_name.strip()
        self.is_preferred_name = is_preferred_name
        self.is_short_name = is_short_name
        self.is_colloquial = is_colloquial
        self.is_historic = is_historic
        self.from_date = from_date.strip()
        self.to_date = to_date.strip()

    def __str__(self):
        return self.alternate_name

    def __repr__(self):
        return f"GeoAlternateName({self.alternate_name_id}, {self.alternate_name})"

    def __eq__(self, other):
        return self.alternate_name_id == other.alternate_name_id

    def __hash__(self):
        return self.alternate_name_id

    @classmethod
    def from_string(cls, line: str):
        alternate_name_id, geoname_id, iso_language, alternate_name, \
        is_preferred_name, is_short_name, is_colloquial, is_historic, \
        from_date, to_date = line.split("\t")

        try:
            is_preferred_name = bool(int(is_preferred_name))
        except ValueError:
            is_preferred_name = False

        try:
            is_short_name = bool(int(is_short_name))
        except ValueError:
            is_short_name = False

        try:
            is_colloquial = bool(int(is_colloquial))
        except ValueError:
            is_colloquial = False

        try:
            is_historic = bool(int(is_historic))
        except ValueError:
            is_historic = False

        return GeoAlternateName(alternate_name_id, geoname_id,
                                iso_language, alternate_name,
                                is_preferred_name, is_short_name,
                                is_colloquial, is_historic, from_date,
                                to_date)


class GeoName(PeopleGroup):

    def __init__(self,
                 geoname_id: str,
                 name: str,
                 ascii_name: str,
                 alternate_names: List[str],
                 latitude: str,
                 longitude: str,
                 feature_class: str,
                 feature_code: str,
                 country_code: str,
                 cc2: List[str],
                 admin1code: str,
                 admin2code: str,
                 admin3code: str,
                 admin4code: str,
                 population: int,
                 elevation: str,
                 dem: str,
                 timezone: str,
                 mod_date: str):
        self.geoname_id = geoname_id.strip()
        self.name = name.strip()
        self.ascii_name = ascii_name.strip()
        self.alternate_names = alternate_names
        self.latitude = latitude.strip()
        self.longitude = longitude.strip()
        self.feature_class = feature_class.strip()
        self.feature_code = feature_code.strip()
        self.country_code = country_code.strip()
        self.cc2 = cc2
        self.admin1code = admin1code.strip()
        self.admin2code = admin2code.strip()
        self.admin3code = admin3code.strip()
        self.admin4code = admin4code.strip()
        self.population = population
        self.elevation = elevation.strip()
        self.dem = dem.strip()
        self.timezone = timezone.strip()
        self.mod_date = mod_date.strip()

        self.geo_alternate_names: List[GeoAlternateName] = []
        self.demonyms: List[str] = []

        self.alias_set = self.update_alias_set()

    def get_name(self) -> str:
        return self.name

    def get_short_name(self) -> str:
        for an in self.geo_alternate_names:
            if an.iso_language == "en" and an.is_preferred_name:
                return an.alternate_name
        return self.get_name()

    def get_id(self) -> str:
        return self.geoname_id

    def get_aliases(self) -> list:
        return list(self.alias_set)

    def has_alias(self, alias: str) -> bool:
        if not alias.strip():
            return False
        return alias.lower().strip() in self.alias_set

    @classmethod
    def from_string(cls, line: str):
        geoname_id, name, ascii_name, alternate_names, latitude, longitude, \
        feature_class, feature_code, country_code, cc2, admin1code, \
        admin2code, admin3code, admin4code, population, elevation, \
        dem, timezone, mod_date = line.split("\t")

        alternate_names = [n.strip() for n in alternate_names.strip().split(",")
                           if n.strip()]
        cc2 = cc2.split(",")

        try:
            population = int(population)
        except ValueError:
            population = 0

        return GeoName(geoname_id, name, ascii_name, alternate_names,
                       latitude, longitude, feature_class, feature_code,
                       country_code, cc2, admin1code, admin2code,
                       admin3code, admin4code, population, elevation,
                       dem, timezone, mod_date)

    def update_alias_set(self):
        self.alias_set = set(
            alias.lower().strip() for alias in
            (
                    [self.name] +
                    [self.ascii_name] +
                    list(self.alternate_names) +
                    [a.alternate_name for a in self.geo_alternate_names if
                     not a.iso_language in {"link", "wkdt"}] +
                    list(self.demonyms) +
                    [demo + "s" for demo in self.demonyms if
                     not (demo.endswith("s") or
                          demo.endswith("ese") or
                          demo.endswith("sh") or
                          demo.endswith("ch") or
                          demo.endswith("x") or
                          demo.endswith("y") or
                          demo.endswith("z") or
                          demo.endswith("man"))] +
                    [demo[:-3] + "men" for demo in self.demonyms if
                     demo.endswith("man")]
            )
        )

        if "Czech" in self.demonyms:
            self.alias_set.update("Czechs")

        return self.alias_set

    def __repr__(self):
        return f"GeoName({self.geoname_id}, {self.name})"
