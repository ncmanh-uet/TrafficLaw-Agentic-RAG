import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PayloadSchemaType,
)

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_SIZE = os.getenv("VECTOR_SIZE")

logger = logging.getLogger(__name__)


def connect_qdrant(
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    timeout: int | None = None,
) -> QdrantClient:
    """Kết nối Qdrant Cloud (url + api_key) hoặc local (host + port)."""
    url = qdrant_url if qdrant_url is not None else QDRANT_URL
    api_key = qdrant_api_key if qdrant_api_key is not None else QDRANT_API_KEY
    host = qdrant_host if qdrant_host is not None else QDRANT_HOST
    port = qdrant_port if qdrant_port is not None else QDRANT_PORT
    t = timeout if timeout is not None else QDRANT_TIMEOUT

    if url:
        logger.info("Kết nối Qdrant Cloud: %s  (timeout=%ss)", url, t)
        return QdrantClient(url=url, api_key=api_key, timeout=t)

    logger.info("Kết nối Qdrant local: %s:%s", host, port)
    return QdrantClient(host=host, port=port, api_key=api_key, timeout=t)


def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    *,
    recreate: bool = False,
) -> None:
    """Tạo collection 768-dim Cosine + payload indexes cho metadata filtering."""
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        if recreate:
            logger.info("Xóa collection cũ: %s", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info(
                "Collection '%s' đã tồn tại. Dùng recreate=True để tạo lại.",
                collection_name,
            )
            return

    logger.info("Tạo collection '%s' (dim=%s, Cosine)", collection_name, VECTOR_SIZE)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    for field, schema in [
        ("doc_id", PayloadSchemaType.KEYWORD),
        ("status", PayloadSchemaType.KEYWORD),
        ("effective_date", PayloadSchemaType.KEYWORD),
        ("topic", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=schema,
        )
        logger.info("  ✓ Payload index: %s (%s)", field, schema)

    logger.info("Collection đã sẵn sàng.\n")
