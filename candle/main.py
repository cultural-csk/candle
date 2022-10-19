import argparse
import logging
from pathlib import Path

from utils.config_reader import read_config

logging.basicConfig(level=logging.INFO,
                    format="[%(processName)s] [%(asctime)s] [%(name)s] "
                           "[%(levelname)s] %(message)s",
                    datefmt='%d-%m %H:%M:%S')

logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="The ethnocultural CSK extraction pipeline")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file",
        required=True)

    parser.add_argument(
        "-p",
        "--people_group",
        choices=["geonames", "religions", "occupations", "regions"],
        help="The people group to use, either \"geonames\" or ...",
        required=True
    )

    parser.add_argument(
        "-s",
        "--spacy_file_list",
        type=str,
        help="Path to the file containing the list of SpaCy files to process",
        required=False
    )

    parser.add_argument(
        "-d",
        "--db_name",
        type=str,
        help="Name of the MongoDB database",
        required=False
    )

    parser.add_argument(
        "-z",
        "--clean_db",
        action="store_true",
        help="Clean the MongoDB database before running the pipeline",
        required=False
    )

    parser.add_argument(
        "-g",
        "--gpus",
        type=str,
        help="The GPUs to use, comma-separated",
        required=False
    )

    parser.add_argument(
        "-i",
        "--components",
        type=int,
        help="The components to run",
        nargs="+"
    )

    parser.add_argument(
        "--cluster_facet",
        type=str,
        help="The domain to cluster",
        required=False
    )

    parser.add_argument(
        "--cluster_nid",
        type=str,
        help="Path to file containing relevant node ids",
        required=False
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Path to the output file",
        required=False
    )

    parser.add_argument(
        "--domain",
        type=str,
        help="The domain name",
        required=False
    )

    args = parser.parse_args()

    logger.info(f"Reading config from {args.config}...")
    config = read_config(args.config)

    # Read file paths and find the absolute paths
    spacy_file_list = []
    if args.spacy_file_list:
        logger.info(f"Reading \"{args.spacy_file_list}\"...")
        with open(args.spacy_file_list, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    file_path = str(Path(line).absolute())
                    spacy_file_list.append(file_path)

    # Overwrite the config
    logger.info(f"Overwriting config with arguments...")
    config["input"] = {
        "spacy_file_list": spacy_file_list,
    }
    config["clean_db"] = False
    if args.clean_db:
        config["clean_db"] = True
    if args.db_name:
        config["mongo_db"]["database"] = args.db_name
    if args.gpus:
        config["gpus"] = [int(g.strip()) for g in args.gpus.split(",") if
                          g.strip()]
    config["chosen_components"] = sorted(args.components)

    if config["clean_db"]:
        logger.info("Cleaning the MongoDB database...")

        prompt = "Are you sure you want to clean the MongoDB database? " \
                 "This will delete all data in the database. [y/N] "
        if input(prompt).lower() == "y":
            from utils.mongodb_handler import get_database
            db = get_database(**config["mongo_db"])
            for collection in db.list_collection_names():
                db.drop_collection(collection)
            logger.info("Done cleaning the MongoDB database.")

        return

    if args.cluster_facet or args.cluster_nid:
        inp = config["pipeline_components"]["clustering_component"].get("input",
                                                                        {})
        if args.cluster_facet:
            inp["label"] = args.cluster_facet
        if args.cluster_nid:
            with open(args.cluster_nid, "r") as f:
                inp["ids"] = [
                    line.strip() for line in f if line.strip()]
        config["pipeline_components"]["clustering_component"]["input"] = inp
        config["pipeline_components"]["representative_generator"]["input"] = inp
        config["pipeline_components"]["ranking_component"]["input"] = inp

        output = config["pipeline_components"]["ranking_component"].get(
            "output", {})
        output["file"] = args.output_file
        config["pipeline_components"]["ranking_component"]["output"] = output

    if args.domain:
        config["domain"] = args.domain

    logger.info("Initializing the pipeline...")
    config["people_group"] = args.people_group
    if args.people_group == "geonames":
        from pipeline.pipeline import GeoNamePipeline
        pipeline = GeoNamePipeline(config)
    elif args.people_group == "religions":
        from pipeline.pipeline import ReligionPipeline
        pipeline = ReligionPipeline(config)
    elif args.people_group == "occupations":
        from pipeline.pipeline import OccupationPipeline
        pipeline = OccupationPipeline(config)
    elif args.people_group == "regions":
        from pipeline.pipeline import RegionPipeline
        pipeline = RegionPipeline(config)
    else:
        return

    pipeline.run()


if __name__ == "__main__":
    main()
