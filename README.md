# CANDLE 🕯: Extracting Cultural Commonsense Knowledge at Scale

## Pipeline execution

There are 6 components
(see [`candle/pipeline/pipeline.py`](candle/pipeline/pipeline.py)):

1. [`candle/pipeline/component_people_group_matcher.py`](candle/pipeline/component_people_group_matcher.py)
2. [`candle/pipeline/component_generic_sentence_filter.py`](candle/pipeline/component_generic_sentence_filter.py)
3. [`candle/pipeline/component_culture_classifier.py`](candle/pipeline/component_culture_classifier.py)
4. [`candle/pipeline/component_clustering.py`](candle/pipeline/component_clustering.py)
5. [`candle/pipeline/component_rep_generator.py`](candle/pipeline/component_rep_generator.py)
6. [`candle/pipeline/component_ranking.py`](candle/pipeline/component_ranking.py)

For example, to run the pipeline for the `religions` domain (see also
[`candle/config_religions.yaml`](candle/config_religions.yaml)), follow these
steps:

Start your local MongoDB instance:

```bash
cd /path/to/mongodb/folder
bin/mongod --dbpath /folder/to/save/the/database --bind_ip_all
```

Run the first 3 components:

```bash
cd candle/candle
python main.py \
  --config config_religions.yaml \
  --people_group religions \
  --spacy_file_list data/spacy/dummy.txt \
  --components 1 2 3
```

Run the last 3 components:

```bash
for facet in "food" "drink" "ritual"
do
  python main.py \
    --config config_religions.yaml \
    --people_group religions \
    --components 4 5 6 \
    --cluster_facet $facet \
    --cluster_nid data/religions/religion_ids.txt \
    --domain religions \
    --output_file _outputs/religions_$facet.jsonl
done
```