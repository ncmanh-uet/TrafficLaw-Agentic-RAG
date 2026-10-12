# -*- coding: utf-8 -*-
"""
indexer_ggcolab.py — Qdrant indexing (Colab / Qdrant Cloud)
======================================================================
Phiên bản rút gọn của indexer.py, chỉ gồm:
  1. Tạo collection (768-dim, Cosine, payload indexes)
  2. Embed + upsert chunks từ all_chunks.jsonl

Chạy trên colab để cải thiện tốc độ.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config (override bằng env vars hoặc tham số run_indexing)
# ---------------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "Traffic_Law_Hybrid")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
VECTOR_SIZE = 768
BATCH_EMBED = int(os.getenv("BATCH_EMBED", "64"))
BATCH_UPSERT = int(os.getenv("BATCH_UPSERT", "100"))

# e5 family requires "passage: " prefix at encode time.
PASSAGE_PREFIX = "passage: " if "e5" in EMBEDDING_MODEL.lower() else ""

# Default JSONL: env > Colab /content > repo Data/
_DEFAULT_JSONL = os.getenv("JSONL_PATH")
if not _DEFAULT_JSONL:
    _colab_path = Path("/content/all_chunks.jsonl")
    _repo_path = Path(__file__).resolve().parent.parent.parent / "Data" / "all_chunks.jsonl"
    JSONL_PATH = _colab_path if _colab_path.exists() else _repo_path
else:
    JSONL_PATH = Path(_DEFAULT_JSONL)

# ---------------------------------------------------------------------------
# Logging (Colab: chỉ stdout; local: thêm file log nếu có thư mục logs)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _setup_logging(log_to_file: bool = True) -> None:
    if logger.handlers:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        try:
            log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime

            log_file = log_dir / f"indexer_colab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        except OSError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_colab_secrets() -> None:
    """Đọc QDRANT_URL / QDRANT_API_KEY từ Colab Secrets nếu chưa set env."""
    global QDRANT_URL, QDRANT_API_KEY

    if QDRANT_URL and QDRANT_API_KEY:
        return

    try:
        from google.colab import userdata  # type: ignore[import-untyped]

        if not QDRANT_URL:
            QDRANT_URL = userdata.get("QDRANT_URL")
        if not QDRANT_API_KEY:
            QDRANT_API_KEY = userdata.get("QDRANT_API_KEY")
    except ImportError:
        pass


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


def index_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    jsonl_path: Path | str = JSONL_PATH,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Đọc all_chunks.jsonl → embed → upsert vào Qdrant theo batch."""
    jsonl_path = Path(jsonl_path)
    logger.info("Đọc dữ liệu từ: %s", jsonl_path)

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file chunks: {jsonl_path}\n"
            "Trên Colab: upload all_chunks.jsonl hoặc set JSONL_PATH / --jsonl."
        )

    chunks: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    total = len(chunks)
    logger.info("Tổng chunks: %s", total)
    if total == 0:
        logger.warning("File JSONL rỗng — bỏ qua indexing.")
        return

    logger.info("Bắt đầu embedding (%s, batch=%s)...", EMBEDDING_MODEL, BATCH_EMBED)
    texts = [PASSAGE_PREFIX + c["content"] for c in chunks]
    all_vectors: list[list[float]] = []

    t0 = time.time()
    for i in range(0, total, BATCH_EMBED):
        batch_texts = texts[i : i + BATCH_EMBED]
        vectors = model.encode(batch_texts, normalize_embeddings=True)
        all_vectors.extend(vectors.tolist())

        done = min(i + BATCH_EMBED, total)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        logger.info("  Embedded %s/%s  (%.1f chunks/s)", done, total, speed)

    embed_time = time.time() - t0
    logger.info("Embedding hoàn tất: %.1fs\n", embed_time)

    logger.info("Upsert vào Qdrant (batch=%s)...", BATCH_UPSERT)
    current_source: str | None = None
    source_count = 0

    t1 = time.time()
    for i in range(0, total, BATCH_UPSERT):
        batch_points: list[PointStruct] = []
        for j in range(i, min(i + BATCH_UPSERT, total)):
            chunk = chunks[j]
            meta = chunk["metadata"]

            src = meta.get("source_file", "")
            if src != current_source:
                if current_source is not None:
                    logger.info("    [%s] → %s points", current_source, source_count)
                current_source = src
                source_count = 0
            source_count += 1

            batch_points.append(
                PointStruct(
                    id=j,
                    vector=all_vectors[j],
                    payload={
                        "chunk_id": meta.get("chunk_id", ""),
                        "doc_id": meta.get("doc_id", ""),
                        "ten_van_ban": meta.get("ten_van_ban", ""),
                        "issuer": meta.get("issuer", ""),
                        "document_type": meta.get("document_type", ""),
                        "status": meta.get("status", "active"),
                        "effective_date": meta.get("effective_date", ""),
                        "topic": meta.get("topic", ""),
                        "dieu": meta.get("dieu", 0),
                        "dieu_title": meta.get("dieu_title", ""),
                        "khoan": meta.get("khoan"),
                        "diem": meta.get("diem"),
                        "level": meta.get("level", 1),
                        "source_file": meta.get("source_file", ""),
                        "token_estimate": meta.get("token_estimate", 0),
                        "content": chunk["content"],
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=batch_points)

    if current_source:
        logger.info("    [%s] → %s points", current_source, source_count)

    upsert_time = time.time() - t1
    logger.info("\nUpsert hoàn tất: %.1fs", upsert_time)

    info = client.get_collection(collection_name)
    logger.info("Points trong Qdrant: %s", info.points_count)
    logger.info("Tổng thời gian: %.1fs\n", embed_time + upsert_time)


def run_indexing(
    *,
    jsonl_path: Path | str | None = None,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
    skip_index: bool = False,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> None:
    """
    Entry point cho Colab notebook.

    Ví dụ:
        run_indexing(recreate=True)
        run_indexing(jsonl_path="/content/drive/MyDrive/all_chunks.jsonl")
    """
    _setup_logging()
    _load_colab_secrets()

    path = Path(jsonl_path) if jsonl_path else JSONL_PATH

    logger.info("=" * 60)
    logger.info("QDRANT INDEXING (no validation)")
    logger.info("=" * 60)

    client = connect_qdrant(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    try:
        client.get_collections()
        logger.info("Kết nối Qdrant thành công!\n")
    except Exception as e:
        logger.error("Không thể kết nối Qdrant: %s", e)
        logger.error("Cloud: set QDRANT_URL + QDRANT_API_KEY (env hoặc Colab Secrets)")
        raise

    logger.info("Loading model: %s", embedding_model)
    model = SentenceTransformer(embedding_model)
    logger.info("Model loaded. Dimension: %s\n", model.get_sentence_embedding_dimension())

    create_collection(client, collection_name, recreate=recreate)

    if not skip_index:
        index_chunks(client, model, path, collection_name)

    logger.info("INDEXING HOÀN TẤT.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index Traffic Law chunks vào Qdrant (không validation)."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Đường dẫn all_chunks.jsonl (mặc định: JSONL_PATH env hoặc auto-detect)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa và tạo lại collection nếu đã tồn tại",
    )
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Chỉ tạo collection, không index chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_indexing(
        jsonl_path=args.jsonl,
        recreate=args.recreate,
        skip_index=args.collection_only,
    )


if __name__ == "__main__":
    main()
