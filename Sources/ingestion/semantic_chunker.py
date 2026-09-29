import os
import re
import yaml
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from .clean_luat import get_base_dir
from .splitter import HierarchicalLegalSplitter, logger
from .utilities import save_chunk_as_md

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


BASE_DIR = get_base_dir()
CLEANED_DIRS = {
    "luat":     BASE_DIR / "Data" / "cleaned" / "luat",
    "nghidinh": BASE_DIR / "Data" / "cleaned" / "nghidinh",
    "thongtu":  BASE_DIR / "Data" / "cleaned" / "thongtu",
}
OUTPUT_DIR  = BASE_DIR / "Data" / "semantic_chunks"
JSONL_PATH  = BASE_DIR / "Data" / "all_chunks.jsonl"   # export cho Qdrant/ChromaDB
LOG_DIR     = BASE_DIR / "logs"



# ---------------------------------------------------------------------------
# Metadata rules — đầy đủ theo prompt (bao gồm effective_date)
# ---------------------------------------------------------------------------
METADATA_RULES: dict[str, dict] = {
    # ── Luật ──────────────────────────────────────────────────────────────
    "luat35_db_2024": {
        "doc_id":         "35/2024/QH15",
        "title":          "Luật Đường bộ 2024",
        "issuer":         "Quốc hội",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Luật nền tảng",
    },
    "luat36_ttatgt_2024": {
        "doc_id":         "36/2024/QH15",
        "title":          "Luật Trật tự ATGT đường bộ 2024",
        "issuer":         "Quốc hội",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Luật nền tảng",
    },

    # ── Nghị định ─────────────────────────────────────────────────────────
    "nd168_2024_XuPhat_TruDiem_DB_baibo_nd100": {
        "doc_id":         "168/2024/NĐ-CP",
        "title":          "Nghị định 168/2024/NĐ-CP (Xử phạt VPHC đường bộ)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Xử phạt vi phạm giao thông",
    },
    "nd160_2024_dtlx": {
        "doc_id":         "160/2024/NĐ-CP",
        "title":          "Nghị định 160/2024/NĐ-CP (Quy định về hoạt động đào tạo và sát hạch lái xe)",
        "issuer":         "Chính phủ",
        "status":         "active",
        "effective_date": "2025-01-01",
        "topic":          "Đào tạo và sát hạch lái xe",
    },

    # ── Thông tư ──────────────────────────────────────────────────────────
    "108_2026_TT-BCA": {
        "doc_id": "108/2026/TT-BCA",
        "title": "QUY ĐỊNH VỀ SÁT HẠCH, CẤP GIẤY PHÉP LÁI XE; CẤP, SỬ DỤNG GIẤY PHÉP LÁI XE QUỐC TẾ",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2026-07-01",
        "topic": "Giấy phép lái xe",
    },
    "13_2025_TT-BCA": {
        "doc_id": "13/2025/TT-BCA",
        "title": "TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ, ĐƯỜNG SẮT VÀ ĐƯỜNG THỦY NỘI ĐỊA",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-03-01",
        "topic": "Quy trình tuần tra, kiểm soát và xử lý vi phạm",
    },
    "155_2025_TT-BTC": {
        "doc_id": "155/2025/TT-BTC",
        "title": "QUY ĐỊNH MỨC THU, CHẾ ĐỘ THU, NỘP, MIỄN LỆ PHÍ ĐĂNG KÝ, CẤP BIỂN PHƯƠNG TIỆN GIAO THÔNG",
        "issuer": "Bộ Tài Chính",
        "status": "active",
        "effective_date": "2026-01-01",
        "topic": "Đăng ký phương tiện",
    },
    "17_2026_TT-BXD": {
        "doc_id": "17/2026/TT-BXD",
        "title": "ĐÀO TẠO LÁI XE; BỒI DƯỠNG, KIỂM TRA, CẤP CHỨNG CHỈ BỒI DƯỠNG KIẾN THỨC PHÁP LUẬT VỀ GIAO THÔNG ĐƯỜNG BỘ",
        "issuer": "Bộ Xây Dựng",
        "status": "active",
        "effective_date": "2026-07-01",
        "topic": "Giấy phép lái xe",
    },
    "35_2024_TT-BGTVT": {
        "doc_id": "35/2024/TT-BGTVT",
        "title": "QUY ĐỊNH VỀ ĐÀO TẠO, SÁT HẠCH, CẤP GIẤY PHÉP LÁI XE; CẤP, SỬ DỤNG GIẤY PHÉP LÁI XE QUỐC TẾ; ĐÀO TẠO, KIỂM TRA, CẤP CHỨNG CHỈ BỒI DƯỠNG KIẾN THỨC PHÁP LUẬT VỀ GIAO THÔNG ĐƯỜNG BỘ",
        "issuer": "Bộ Giao Thông Vận Tải",
        "status": "active",
        "effective_date": "2025-01-01",
        "topic": "Giấy phép lái xe",
    },
    "51_2025_TT-BCA": {
        "doc_id": "51/2025/TT-BCA",
        "title": "QUY ĐỊNH VỀ CẤP, THU HỒI CHỨNG NHẬN ĐĂNG KÝ XE, BIỂN SỐ XE CƠ GIỚI, XE MÁY CHUYÊN DÙNG",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-07-01",
        "topic": "Đăng ký phương tiện",
    },
    "67_2024_TT-BCA": {
        "doc_id": "67/2024/TT-BCA",
        "title": "QUY ĐỊNH QUY TRÌNH QUẢN LÝ, SỬ DỤNG PHƯƠNG TIỆN, THIẾT BỊ KỸ THUẬT NGHIỆP VỤ TRONG CÔNG AN NHÂN DÂN VÀ DỮ LIỆU THU ĐƯỢC TỪ PHƯƠNG TIỆN, THIẾT BỊ KỸ THUẬT DO CÁ NHÂN, TỔ CHỨC CUNG CẤP ĐỂ PHÁT HIỆN VI PHẠM HÀNH CHÍNH",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-01-01",
        "topic": "Quy trình tuần tra, kiểm soát và xử lý vi phạm",
    },
    "72_2024_TT-BCA": {
        "doc_id": "72/2024/TT-BCA",
        "title": "QUY ĐỊNH QUY TRÌNH ĐIỀU TRA, GIẢI QUYẾT TAI NẠN GIAO THÔNG ĐƯỜNG BỘ CỦA CẢNH SÁT GIAO THÔNG",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-01-01",
        "topic": "Quy trình tuần tra, kiểm soát và xử lý vi phạm",
    },
    "73_2024_TT-BCA": {
        "doc_id": "73/2024/TT-BCA",
        "title": "QUY ĐỊNH CÔNG TÁC TUẦN TRA, KIỂM SOÁT, XỬ LÝ VI PHẠM PHÁP LUẬT VỀ TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ CỦA CẢNH SÁT GIAO THÔNG",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-01-01",
        "topic": "Quy trình tuần tra, kiểm soát và xử lý vi phạm",
    },
    "79_2024_TT-BCA": {
        "doc_id": "79/2024/TT-BCA",
        "title": "QUY ĐỊNH VỀ CẤP, THU HỒI CHỨNG NHẬN ĐĂNG KÝ XE, BIỂN SỐ XE CƠ GIỚI, XE MÁY CHUYÊN DÙNG",
        "issuer": "Bộ Công An",
        "status": "active",
        "effective_date": "2025-01-01",
        "topic": "Đăng ký phương tiện",
    },
}

def get_file_metadata(file_stem: str) -> dict:
    """Lấy metadata dựa trên tên file (không có extension)."""
    meta = METADATA_RULES.get(file_stem)
    if meta:
        return meta
    # Fallback thông minh: tìm key substring match
    for key, val in METADATA_RULES.items():
        if key.lower() in file_stem.lower() or file_stem.lower() in key.lower():
            logger.debug(f"Metadata fuzzy match: {file_stem} → {key}")
            return val
    logger.warning(f"Không tìm thấy metadata rule cho: {file_stem}")
    return {
        "doc_id":         file_stem,
        "title":          file_stem,
        "issuer":         "Không xác định",
        "status":         "active",
        "effective_date": "Không xác định",
        "topic":          "Chung",
    }



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splitter     = HierarchicalLegalSplitter()
    all_chunks   = []
    global_stats = {
        "total_files": 0,
        "total_chunks": 0,
        "duplicates_removed": 0,
        "by_doc_type": {},
        "by_level": {1: 0, 2: 0, 3: 0},
    }

    for doc_type, dir_path in CLEANED_DIRS.items():
        if not dir_path.exists():
            logger.warning(f"Thư mục không tồn tại: {dir_path}")
            continue

        md_files = list(dir_path.rglob("*.md"))
        # Bỏ qua file stats
        md_files = [f for f in md_files if not f.name.startswith("_")]

        logger.info(f"\n[{doc_type.upper()}] Tìm thấy {len(md_files)} file")
        global_stats["by_doc_type"][doc_type] = {"files": len(md_files), "chunks": 0}

        for file_path in tqdm(md_files, desc=f"Chunking {doc_type}", unit="file"):
            global_stats["total_files"] += 1

            doc_meta = get_file_metadata(file_path.stem)

            # ── FIX output path collision: dùng doc_type + relative path ──
            rel      = file_path.relative_to(dir_path)
            out_subdir = OUTPUT_DIR / doc_type / rel.parent / file_path.stem
            out_subdir.mkdir(parents=True, exist_ok=True)

            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            if not full_text.strip():
                logger.warning(f"  File rỗng: {file_path.name}")
                continue

            chunks = splitter.chunk_document(
                full_text   = full_text,
                doc_meta    = doc_meta,
                doc_type    = doc_type,
                source_file = file_path.name,
            )

            for chunk in chunks:
                save_chunk_as_md(chunk, out_subdir)
                all_chunks.append(chunk)
                lvl = chunk["metadata"]["level"]
                global_stats["by_level"][lvl] = global_stats["by_level"].get(lvl, 0) + 1

            global_stats["total_chunks"]                         += len(chunks)
            global_stats["by_doc_type"][doc_type]["chunks"]      += len(chunks)

            logger.info(
                f"  {file_path.name} → {len(chunks)} chunks "
                f"(L1:{sum(1 for c in chunks if c['metadata']['level']==1)} "
                f"L2:{sum(1 for c in chunks if c['metadata']['level']==2)} "
                f"L3:{sum(1 for c in chunks if c['metadata']['level']==3)})"
            )

    # ── Export tất cả chunks ra JSONL (dùng để load vào ChromaDB / Qdrant) ──
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"TỔNG KẾT CHUNKING:")
    logger.info(f"  Tổng file:          {global_stats['total_files']}")
    logger.info(f"  Tổng chunks:        {global_stats['total_chunks']}")
    logger.info(f"  Theo level:         L1={global_stats['by_level'].get(1,0)} | "
                f"L2={global_stats['by_level'].get(2,0)} | L3={global_stats['by_level'].get(3,0)}")
    for dt, s in global_stats["by_doc_type"].items():
        logger.info(f"  [{dt}]: {s['files']} files → {s['chunks']} chunks")
    logger.info(f"  JSONL export: {JSONL_PATH}")

    # Lưu stats JSON
    stats_path = OUTPUT_DIR / "_chunking_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)

    logger.info("HOÀN TẤT.")
    return all_chunks


if __name__ == "__main__":
    main()