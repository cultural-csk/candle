# CANDLE 🕯: Extracting Cultural Commonsense Knowledge at Scale


[![CANDLE Introduction Video](https://img.youtube.com/vi/F4eElgjE4R8/0.jpg)](https://www.youtube.com/watch?v=F4eElgjE4R8)


## Running spaCy on your input corpus

The first step is to run spaCy on your input corpus of choice. The
script [`candle/run_spacy.py`](candle/run_spacy.py) can be used for this
purpose. For example, to run this script on the dummy files in
the [`candle/data/input_corpus`](candle/data/input_corpus) directory,
run the following command:

```bash
cd candle
python run_spacy.py \
    -i data/input_corpus/dummy-000.jsonl \
    -o data/spacy/dummy-000.spacy
```

The input file should be a JSONL file, where each line is a JSON object
with the following fields:

- `text`: The text of the document (required).
- `timestamp`: The timestamp of the document (optional).
- `url`: The URL of the document (optional).

After running spaCy on all the input files, you should create a file consisting
of the paths to all the spaCy output files (see
e.g., [`candle/data/spacy/dummy.txt`](candle/data/spacy/dummy.txt)).
This file should be passed to the next steps using the `spacy_file_list`
argument (see below).

## CANDLE pipeline execution

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

**Start your local MongoDB instance:**

```bash
cd /path/to/mongodb/folder
bin/mongod --dbpath /folder/to/save/the/database --bind_ip_all
```

**Run the first 3 components:**

```bash
cd candle/candle
python main.py \
  --config config_religions.yaml \
  --people_group religions \
  --spacy_file_list data/spacy/dummy.txt \
  --components 1 2 3
```

**Run the last 3 components:**

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
