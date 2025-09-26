from src.rag.ingest import seed_if_empty
from src.ui.gradio_app import build
from src.config import SERVER_PORT
from src.schema import kb_schema, cache_schema
from redisvl.index import SearchIndex
from src.db.redis_client import r
from src.routing.semantic_router import init_router   # ⬅️ add this
from src.memory.long_term import migrate_legacy_array_if_present
from src.config import DEFAULT_USER_ID

# ensure indexes exist at startup
for schema in (kb_schema, cache_schema):
    idx = SearchIndex.from_dict(schema, client=r)
    if not idx.exists():
        idx.create(overwrite=False)

seed_if_empty()
init_router(overwrite=False)
migrate_legacy_array_if_present(DEFAULT_USER_ID)

if __name__ == "__main__":
    app = build()
    app.launch(server_name="0.0.0.0", server_port=SERVER_PORT)