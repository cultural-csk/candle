import pymongo

cached_db = {
}


def get_database(host: str, port: int, database: str):
    """
    Returns a pymongo database object
    :param host: MongoDB host
    :param port: MongoDB port
    :param database: MongoDB database
    :return: pymongo database object
    """
    if (host, port, database) not in cached_db:
        cached_db[(host, port, database)] = pymongo.MongoClient(host, port)[
            database]

    return cached_db[(host, port, database)]
