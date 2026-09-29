"""Làm sạch PDF nghị định theo cùng luồng xử lý với ``clean_luat.py``."""

import json
import logging
from datetime import datetime

import pdfplumber
from tqdm import tqdm


from clean_luat import build_page_blocks, clean_text, get_table_bboxes
from .utilities import get_base_dir

BASE_DIR = get_base_dir()
RAW_NGHIDINH_DIR = BASE_DIR / "Data" / "raw" / "nghidinh"
CLEANED_NGHIDINH_DIR = BASE_DIR / "Data" / "cleaned" / "nghidinh"
LOG_DIR = BASE_DIR / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"clean_nghidinh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def process_decree_pdfs() -> dict:
    """Trích text/bảng theo vị trí đọc, làm sạch, rồi ghi một Markdown mỗi PDF.

    Không lọc theo Điều, keyword hay ``FILE_STRATEGIES``: toàn bộ nội dung của
    mỗi nghị định trong thư mục raw đều được giữ lại.
    """
    CLEANED_NGHIDINH_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(RAW_NGHIDINH_DIR.glob("*.pdf"))
    logger.info("Tìm thấy %d file nghị định trong %s", len(pdf_files), RAW_NGHIDINH_DIR)

    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "errors": 0,
        "files": {},
    }

    for pdf_path in tqdm(pdf_files, desc="Xử lý file Nghị định", unit="file"):
        full_content: list[str] = []
        page_count = 0
        table_count = 0
        char_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for page in tqdm(pdf.pages, desc=f"  Trang {pdf_path.stem}", leave=False):
                    # Giống clean_luat.py: lấy bbox trước để loại text trong
                    # bảng và ghép text/bảng theo đúng thứ tự từ trên xuống.
                    table_bboxes = get_table_bboxes(page)
                    page_blocks = build_page_blocks(page, table_bboxes, strategy=None)

                    for block in page_blocks:
                        if block["type"] == "text":
                            cleaned = clean_text(block["content"])
                            if cleaned:
                                full_content.append(cleaned)
                                char_count += len(cleaned)
                        else:  # block["type"] == "table"
                            full_content.append("\n\n" + block["content"] + "\n\n")
                            table_count += 1

            output_filename = pdf_path.stem + ".md"
            output_path = CLEANED_NGHIDINH_DIR / output_filename
            output_path.write_text("\n\n".join(full_content), encoding="utf-8")

            stats["processed"] += 1
            stats["files"][pdf_path.name] = {
                "pages": page_count,
                "tables": table_count,
                "chars": char_count,
                "output": str(output_path),
            }
            logger.info(
                "  ✓ Hoàn tất: %s | %d trang | %d bảng | %s ký tự",
                output_filename,
                page_count,
                table_count,
                f"{char_count:,}",
            )
        except Exception as error:
            logger.error("  ✗ Lỗi khi xử lý %s: %s", pdf_path.name, error, exc_info=True)
            stats["errors"] += 1

    stats_path = CLEANED_NGHIDINH_DIR / "_processing_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Stats đã lưu tại: %s", stats_path)
    return stats


if __name__ == "__main__":
    process_decree_pdfs()
