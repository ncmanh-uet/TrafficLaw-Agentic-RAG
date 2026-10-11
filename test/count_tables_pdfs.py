"""Đếm số bảng có cấu trúc trong một PDF đặt trong thư mục test.

Chạy:
    python test\\count_tables_pdfs.py
    python test\\count_tables_pdfs.py ten_file.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
INGESTION_DIR = TEST_DIR.parent / "Data" / "raw" / "thongtu"
sys.path.insert(0, str(INGESTION_DIR))

# Dùng cùng điều kiện nhận diện bảng với pipeline ingestion: chỉ đếm bảng có
# cấu trúc, không đếm các khung/chữ nhật tình cờ bị PDF nhận nhầm là bảng.
from Sources.ingestion.clean_luat import get_table_bboxes  # noqa: E402


def resolve_pdf(filename: str | None) -> Path:
    """Trả về PDF cần đếm; tự chọn khi test chỉ có đúng một PDF."""
    if filename:
        pdf_path = INGESTION_DIR / filename
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")
        return pdf_path

    pdf_files = sorted(TEST_DIR.glob("*.pdf"))
    if len(pdf_files) == 1:
        return pdf_files[0]
    if not pdf_files:
        raise FileNotFoundError(
            f"Không có file PDF nào trong {TEST_DIR}. "
            "Hãy chép một PDF vào đây hoặc truyền tên file."
        )
    names = ", ".join(pdf.name for pdf in pdf_files)
    raise ValueError(f"Có nhiều PDF trong {TEST_DIR}: {names}. Hãy truyền tên file cần đếm.")


def count_tables(pdf_path: Path) -> tuple[int, list[int]]:
    """Trả về tổng số bảng và số bảng tương ứng từng trang."""
    import pdfplumber

    tables_per_page: list[int] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables_per_page.append(len(get_table_bboxes(page)))
    return sum(tables_per_page), tables_per_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Đếm bảng có cấu trúc trong PDF ở thư mục test.")
    parser.add_argument("filename", nargs="?", help="Tên file PDF trong thư mục test")
    args = parser.parse_args()

    pdf_path = resolve_pdf(args.filename)
    total, tables_per_page = count_tables(pdf_path)

    print(f"File: {pdf_path.name}")
    print(f"Số trang: {len(tables_per_page)}")
    for page_number, table_count in enumerate(tables_per_page, start=1):
        if table_count:
            print(f"  Trang {page_number}: {table_count} bảng")
    print(f"Tổng số bảng: {total}")


if __name__ == "__main__":
    main()
