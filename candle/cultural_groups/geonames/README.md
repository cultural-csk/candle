# Parsing data from geonames.org into a tree structure

Getting the data from http://www.geonames.org and appending it with demonyms
from
[this GitHub repo](https://github.com/dok/demonyms/blob/master/demonyms.json) 
(looks like it was crawled from Wikipedia).

Data from geonames.org is quite clean but not perfect.
Some processing steps are necessary to get to a decent result.

## Processing geonames data

- Root node: Earth, ``geonameid = 6295630`` (https://www.geonames.org/6295630)
- Only the following feature codes are extracted from the geonames.org database:

```python
interested_feature_codes = {
    "CONT",  # continent
    "PCL", "PCLD", "PCLF", "PCLH", "PCLI", "PCLIX", "PCLS",  # country
    "ADM1", "ADM2", "ADM3",  # administrative division
    "PPLA", "PPLA2", "PPLAG", "PPLC", "PPL",  # populated place
    # "RGN", "RGNE",  # region
}
```

- ``PPL`` entities must have a population of at least 1000.
- For each child node, only the first found parent-child relation in
  the ``hierarchy.txt`` file is used to build the tree.
- Add more hierarchy information by using admin1code and admin2code.
- If the direct parent is not found, use the closest parent in the tree
  (max_tries = 10 to avoid infinite loop). If no
  parent is found, use the root node. However, all 1st-level nodes must be
  continents (``CONT``).
- Nodes with no parent are removed from the tree.

## Alternate names

Use the alternate names from the geonames.org database.

## Mapping places from ``demonyms.json`` to the GeoNames tree

- Filter the nodes with the place name.
- Get the node with the largest population. Tie-break by using the
  lowest node level.

## Concatenated aliases

The final list of aliases is built by concatenating the following:

- Geonames name.
- Ascii name.
- Alternate names in the ``allCountries.txt`` file.
- Alternate names in the ``alternateNamesV2.txt`` file (excluding ``link``
  and ``wkdt``).
- Demonyms and their plural forms.

## Results

- The final tree has 277,467 nodes.
- 898 nodes has demonyms.
- There are 1,302,065 aliases in the final list.
- In average, each node has about 4.7 aliases.