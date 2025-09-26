from src.config import EMBED_DIM

KB_INDEX_NAME = "idx:kb"
KB_PREFIX = "kb:doc:chunk:"

CACHE_INDEX_NAME = "idx:cache"
CACHE_PREFIX = "cache:qa:"

kb_schema = {
    "index": {"name": KB_INDEX_NAME, "prefix": KB_PREFIX, "storage_type": "json"},
    "fields": [
        {"name": "doc_id", "type": "tag", "path": "$.doc_id"},
        {"name": "chunk_id", "type": "tag", "path": "$.chunk_id"},
        {"name": "chunk_index", "type": "numeric", "path": "$.chunk_index"},
        {"name": "text", "type": "text", "path": "$.text"},
        {"name": "source", "type": "text", "path": "$.source"},
        {"name": "file_name", "type": "text", "path": "$.file_name"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": EMBED_DIM,
                "distance_metric": "cosine",
                "datatype": "float32",
                "path": "$.embedding",
            },
        },
    ],
}

cache_schema = {
    "index": {"name": CACHE_INDEX_NAME, "prefix": CACHE_PREFIX, "storage_type": "json"},
    "fields": [
        {"name": "qa_id", "type": "tag", "path": "$.qa_id"},
        {"name": "prompt", "type": "text", "path": "$.prompt"},
        {"name": "generic_answer", "type": "text", "path": "$.generic_answer"},
        {"name": "meta", "type": "text", "path": "$.meta"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": EMBED_DIM,
                "distance_metric": "cosine",
                "datatype": "float32",
                "path": "$.embedding",
            },
        },
    ],
}