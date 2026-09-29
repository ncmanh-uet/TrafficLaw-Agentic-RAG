import os
import re
import unicodedata
import logging
import json
from pathlib import Path
from datetime import datetime
import pdfplumber
from tqdm import tqdm
from .utilities import get_base_dir

BASE_DIR = get_base_dir()
RAW_LUAT_DIR   = BASE_DIR / "Data" / "raw"    / "luat"
CLEANED_LUAT_DIR = BASE_DIR / "Data" / "cleaned" / "luat"
LOG_DIR        = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"clean_luat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Blacklist — các file KHÔNG xử lý
# ---------------------------------------------------------------------------
EXCLUDED_FILES = {
    "nd17_2026_sua_doi_bo_sung.pdf",   # Hàng không — ngoài phạm vi
    "nd81_2026_XuPhat_DuongSat.pdf",   # Đường sắt — ngoài phạm vi
    "TT130_2025_BTC_LePhi_CapBang.pdf",# Bộ Quốc phòng — ngoài phạm vi
    "luat23_db_2008.pdf",              # Luật 2008 đã hết hiệu lực — dùng bản inactive riêng
}

# ---------------------------------------------------------------------------
# Per-file strategy — logic xử lý riêng từng văn bản
# ---------------------------------------------------------------------------
FILE_STRATEGIES = {
    # Luật 35 & 36 giữ toàn bộ
    "luat35_db_2024.pdf":     {"keep_all": True},
    "luat36_ttatgt_2024.pdf": {"keep_all": True},
    # Luật 2008: giữ nhưng gán inactive, chỉ dùng tra cứu lịch sử
    "luat23_db_2008_inactive.pdf": {
        "keep_all": True,
        "prepend_warning": (
            "⚠️ LƯU Ý: VĂN BẢN NÀY ĐÃ HẾT HIỆU LỰC KỂ TỪ 01/01/2025.\n"
            "Luật Giao thông đường bộ 2008 được thay thế hoàn toàn bởi:\n"
            "  • Luật Đường bộ 35/2024/QH15\n"
            "  • Luật Trật tự ATGT đường bộ 36/2024/QH15\n"
            "Chỉ sử dụng văn bản này cho mục đích tra cứu lịch sử pháp lý.\n"
        )
    },
}

# ---------------------------------------------------------------------------
# Regex làm sạch
# ---------------------------------------------------------------------------
RE_HEADER_NOISE  = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4},.*about:blank$",     re.MULTILINE)
RE_FOOTER_NOISE  = re.compile(r"about:blank\s+\d+/\d+|Thư viện pháp luật|Mã tra cứu", re.IGNORECASE)
RE_PAGE_NUM      = re.compile(r"^(Trang\s+)?\d+(\s*/\s*\d+)?$",               re.IGNORECASE)
RE_FORM_DOTS     = re.compile(r"(\.{5,}|_{5,})")
RE_CHECKBOX      = re.compile(r"([☐☑\uf06f])")

# Phát hiện tiêu đề cấu trúc — KHÔNG được nối với dòng trên
RE_IS_HEADING    = re.compile(
    r"^(Điều\s+\d+|Khoản\s+\d+|Chương\s+[IVXLCDM]+|Phần\s+[IVXLCDM]+|"
    r"\d+\.\s+[A-ZĐÀÁẠẢÃ]|[a-z]\)\s+)",
    re.IGNORECASE
)

# Ký tự kết thúc câu — dòng kết thúc bằng những ký tự này mới được nối
RE_MERGE_ELIGIBLE_END = re.compile(
    r'[a-zA-ZáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ,;]$'
)

# ---------------------------------------------------------------------------
# Hàm làm sạch text
# ---------------------------------------------------------------------------
def clean_text(raw_text: str, strategy: dict | None = None) -> str:
    """
    Làm sạch text: loại nhiễu, định dạng phân cấp, nối dòng bị gãy.
    Không nối nhầm dòng tiêu đề Điều/Khoản vào dòng trước.
    """
    if not raw_text:
        return ""

    text = unicodedata.normalize("NFC", raw_text)
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # --- Lọc nhiễu ---
        if RE_HEADER_NOISE.match(line):   continue
        if RE_FOOTER_NOISE.search(line):  continue
        if RE_PAGE_NUM.match(line):        continue

        # --- Lọc biểu mẫu ---
        line = RE_FORM_DOTS.sub(" [Cần điền thông tin] ", line)
        line = RE_CHECKBOX.sub(" [Lựa chọn] ", line).strip()
        if line in ("", "[Cần điền thông tin]", "[Lựa chọn]"):
            continue

        # --- Định dạng phân cấp Markdown ---
        line = re.sub(
            r"^(Chương\s+[IVXLCDM]+.*|Phần\s+[IVXLCDM]+.*)$",
            r"# \1", line, flags=re.IGNORECASE
        )
        line = re.sub(
            r"^(Điều\s+\d+[.:].*|## Điều\s+\d+[.:].*)",
            lambda m: "## " + m.group(0).lstrip("# "),
            line, flags=re.IGNORECASE
        )

        # --- Bôi đậm ngày tháng ---
        line = re.sub(
            r"(ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
            r"**\1**", line, flags=re.IGNORECASE
        )

        # --- LaTeX cho công thức kỹ thuật ---
        line = re.sub(r"\b(CO2?|HC|NOx|PM2\.5)\b", r"$$\1$$", line)

        cleaned_lines.append(line)

    # --- Nối dòng bị gãy (cải tiến: không nối tiêu đề) ---
    merged_lines = []
    for line in cleaned_lines:
        if not merged_lines:
            merged_lines.append(line)
            continue

        prev = merged_lines[-1]
        prev_ends_mid_sentence = RE_MERGE_ELIGIBLE_END.search(prev[-1:]) if prev else False
        curr_is_heading        = RE_IS_HEADING.match(line)
        curr_starts_lowercase  = (
            line[0].islower() or
            line[0] in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        )

        if prev_ends_mid_sentence and curr_starts_lowercase and not curr_is_heading:
            merged_lines[-1] = prev + " " + line
        else:
            merged_lines.append(line)

    result = "\n".join(merged_lines)

    # Thêm warning nếu strategy yêu cầu
    if strategy and strategy.get("prepend_warning"):
        result = f"> {strategy['prepend_warning']}\n\n{result}"

    return result


# ---------------------------------------------------------------------------
# Chuyển bảng sang Markdown (với flatten merged cells)
# ---------------------------------------------------------------------------
def _normalise_table_data(table_data: list) -> list[list[str]]:
    """Chuẩn hoá dữ liệu bảng và gộp các dòng bị quấn trong cùng một hàng.

    `pdfplumber` thường trả về mỗi dòng hiển thị của một ô cao thành một row.
    Trong các PDF luật, dấu hiệu đáng tin cậy của một dòng quấn là ô đầu tiên
    trống, trong khi row ngay trước đó có mã/số thứ tự.  Gộp theo từng cột ở
    đây giúp giữ nguyên thứ tự thông tin trong hàng thay vì biến nó thành các
    hàng Markdown độc lập.
    """
    rows = [
        [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        for row in table_data
        if any(cell and str(cell).strip() for cell in row)
    ]
    if not rows:
        return []

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    merged: list[list[str]] = []
    for row in rows:
        # Không có giá trị ở cột neo đầu tiên => đây là phần tiếp của row trước.
        if merged and not row[0]:
            previous = merged[-1]
            for column, value in enumerate(row):
                if value:
                    previous[column] = f"{previous[column]} {value}".strip()
        else:
            merged.append(row)
    return merged


def is_structured_table(table_data: list) -> bool:
    """Chỉ chấp nhận bảng có cấu trúc nhiều cột lặp lại.

    Một số PDF tạo rectangle cho từng dòng chữ. Với cấu hình `lines`, chúng
    dễ bị nhận nhầm là bảng lớn. Bảng thật không chỉ có nhiều cột ở vài dòng
    riêng lẻ, mà phải có tỷ lệ đáng kể các hàng chứa đồng thời từ hai ô dữ liệu
    trở lên. Điều này loại các biểu mẫu/văn bản có một đoạn vô tình bị chia cột.
    """
    rows = _normalise_table_data(table_data)
    if len(rows) < 2 or not rows:
        return False

    populated_columns = {
        index
        for row in rows
        for index, cell in enumerate(row)
        if cell
    }
    multi_column_rows = sum(sum(bool(cell) for cell in row) >= 2 for row in rows)
    multi_column_ratio = multi_column_rows / len(rows)
    return (
        len(populated_columns) >= 2
        and multi_column_rows >= 2
        and multi_column_ratio >= 0.25
    )


def table_to_markdown(table_data: list) -> str:
    """Chuyển bảng đã chuẩn hoá thành Markdown, giữ nội dung mỗi hàng liền mạch."""
    if not table_data:
        return ""

    table_data = _normalise_table_data(table_data)
    if not table_data:
        return ""

    max_cols = max(len(row) for row in table_data)

    last_seen = [""] * max_cols
    flattened = []

    for row in table_data:
        new_row = []
        for j, cell in enumerate(row):
            cell_str = cell.strip() if cell else ""
            if not cell_str and last_seen[j]:
                cell_str = last_seen[j]   # flatten merged cell
            elif cell_str:
                last_seen[j] = cell_str
            new_row.append(cell_str)
        flattened.append(new_row)

    md_lines = []
    for i, row in enumerate(flattened):
        # Escape pipe trong nội dung ô
        escaped = [cell.replace("|", "\\|") for cell in row]
        md_lines.append("| " + " | ".join(escaped) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Lấy bbox của tất cả bảng trên một trang
# ---------------------------------------------------------------------------
def get_table_bboxes(page) -> list[tuple]:
    """Trả về bbox của các bảng thật, không phải các dòng văn bản có khung."""
    try:
        return [
            table.bbox
            for table in page.find_tables(table_settings=STRICT_TABLE_SETTINGS)
            if is_structured_table(table.extract())
        ]
    except Exception:
        return []


def is_inside_table(obj: dict, bboxes: list[tuple], tolerance: float = 2.0) -> bool:
    """Kiểm tra obj có nằm trong vùng bảng không."""
    x0 = obj.get("x0", 0)
    top = obj.get("top", 0)
    for (tx0, ttop, tx1, tbottom) in bboxes:
        if (tx0 - tolerance <= x0 and
                x0 <= tx1 + tolerance and
                ttop - tolerance <= top and
                top <= tbottom + tolerance):
            return True
    return False


# ---------------------------------------------------------------------------
# Cấu hình table_settings dùng chung cho extract_tables()
# (trước đây khai báo cục bộ bên trong vòng lặp — giữ nguyên giá trị,
#  chỉ nâng lên module-level để build_page_blocks() cũng dùng được)
# ---------------------------------------------------------------------------
STRICT_TABLE_SETTINGS = {
    "vertical_strategy":   "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance":      3,
    "join_tolerance":      3,
}


# ---------------------------------------------------------------------------
# Ghép text + table theo vị trí đọc (top -> bottom) trên từng trang
# ---------------------------------------------------------------------------
def build_page_blocks(page, table_bboxes: list[tuple], strategy: dict | None) -> list[dict]:
    """
    Trả về danh sách block (text/table) của MỘT trang, theo đúng thứ tự đọc
    (top -> bottom), thay vì gộp toàn bộ text trước rồi mới nối bảng vào cuối.

    Nguyên lý:
      - Dùng bbox của các bảng (đã có sẵn từ get_table_bboxes) để chia trang
        thành các "dải" (band) theo trục dọc: dải text (trước / giữa / sau
        bảng) và dải bảng.
      - Với MỖI dải, tái sử dụng nguyên vẹn `extract_text()` / `extract_tables()`
        + `table_to_markdown()` hiện có — chỉ giới hạn vùng trích xuất theo
        bbox của dải (page.crop). Vì vậy hành vi trích xuất text/table không
        đổi so với trước, chỉ thay đổi cách các block được XẾP THỨ TỰ và GHÉP
        lại với nhau.
      - Việc loại text nằm trong vùng bảng vẫn dùng `is_inside_table()` như
        code gốc, áp dụng cho từng dải text.
      - `strategy` (vd: prepend_warning) chỉ được áp dụng đúng MỘT LẦN cho
        mỗi trang — giống hệt số lần nó được áp dụng trong code gốc (code
        gốc gọi clean_text() đúng 1 lần/trang) — để tránh lặp lại cảnh báo
        nhiều lần chỉ vì một trang giờ bị chia thành nhiều dải text nhỏ hơn.

    Mỗi block trả về có dạng:
        {"type": "text" | "table", "top": float, "content": str, "strategy": dict|None}
    ("strategy" luôn None với block table; chỉ có ý nghĩa với block text vì
    strategy hiện tại chỉ ảnh hưởng tới clean_text()).

    Giới hạn đã biết (xem thêm trong báo cáo refactor):
      - PDF nhiều cột: mỗi dải vẫn dùng đúng thuật toán reading-order mặc
        định của extract_text() (giống hệt hành vi hiện tại trên toàn
        trang) — không có layout engine multi-column riêng, nên không tệ
        hơn nhưng cũng không tốt hơn code gốc với multi-column.
      - Nếu 2+ bảng nằm cạnh nhau theo chiều ngang (bbox chồng lấn theo
        top/bottom), chúng bị gộp vào cùng một "table band" và xuất ra theo
        đúng thứ tự mà extract_tables() trả về cho vùng đó — không đảm bảo
        tuyệt đối thứ tự đọc trái->phải trong layout bảng phức tạp.
      - Text nằm rất sát mép bảng (trong khoảng đệm nhỏ ~2pt) có thể bị
        gộp vào dải bên cạnh thay vì dải "đúng" của nó về mặt hình thức.
    """
    PAD = 2.0
    width, height = page.width, page.height

    # Không có bảng -> giữ nguyên hành vi cũ y hệt (extract_text() toàn trang)
    if not table_bboxes:
        text = page.extract_text()
        return [{"type": "text", "top": 0.0, "content": text, "strategy": strategy}] if text else []

    # Gộp các bbox bảng chồng lấn / liền kề theo chiều dọc thành 1 "table band"
    # (trường hợp bảng nằm cạnh nhau theo chiều ngang trên cùng khoảng top/bottom)
    sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])  # sort theo top
    band_tables: list[list[tuple]] = []
    band_range: list[list[float]] = []  # [top, bottom]
    for bbox in sorted_bboxes:
        _, top, _, bottom = bbox
        if band_range and top <= band_range[-1][1] + PAD:
            band_range[-1][1] = max(band_range[-1][1], bottom)
            band_tables[-1].append(bbox)
        else:
            band_range.append([top, bottom])
            band_tables.append([bbox])

    def _extract_text_band(y0: float, y1: float) -> str | None:
        """Trích text trong dải [y0, y1), loại bỏ nội dung nằm trong bảng."""
        y0c, y1c = max(0.0, y0), min(height, y1)
        if y1c - y0c < 0.5:
            return None
        try:
            cropped = page.crop((0, y0c, width, y1c), relative=False, strict=False)
        except Exception as e:
            logger.debug(f"    Không thể crop dải text ({y0c:.1f}-{y1c:.1f}): {e}")
            return None
        filtered = cropped.filter(lambda obj: not is_inside_table(obj, table_bboxes))
        return filtered.extract_text()

    blocks: list[dict] = []
    cursor = 0.0
    strategy_applied = False  # đảm bảo strategy chỉ áp dụng 1 lần/trang

    for (band_top, band_bottom), bboxes_in_band in zip(band_range, band_tables):
        # --- Dải text trước bảng (nếu có) ---
        text = _extract_text_band(cursor, band_top)
        if text:
            blocks.append({
                "type": "text",
                "top": cursor,
                "content": text,
                "strategy": strategy if not strategy_applied else None,
            })
            strategy_applied = True

        # --- Dải bảng: trích tables trong đúng vùng này ---
        try:
            table_crop = page.crop(
                (0, max(0.0, band_top - PAD), width, min(height, band_bottom + PAD)),
                relative=False, strict=False,
            )
            extracted_tables = table_crop.extract_tables(table_settings=STRICT_TABLE_SETTINGS)
        except Exception as e:
            logger.debug(f"    Không thể trích bảng dải ({band_top:.1f}-{band_bottom:.1f}): {e}")
            extracted_tables = []

        for table in extracted_tables:
            # Crop có thể sinh thêm "bảng" giả từ rectangle của text; xác nhận
            # lại cấu trúc trước khi loại text trong vùng này và xuất Markdown.
            if not is_structured_table(table):
                continue
            md_table = table_to_markdown(table)
            if md_table:
                blocks.append({"type": "table", "top": band_top, "content": md_table, "strategy": None})

        cursor = band_bottom

    # --- Dải text cuối cùng, sau bảng cuối ---
    text = _extract_text_band(cursor, height)
    if text:
        blocks.append({
            "type": "text",
            "top": cursor,
            "content": text,
            "strategy": strategy if not strategy_applied else None,
        })

    return blocks


# ---------------------------------------------------------------------------
# Hàm xử lý chính
# ---------------------------------------------------------------------------
def process_law_pdfs():
    CLEANED_LUAT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_LUAT_DIR.glob("*.pdf"))
    logger.info(f"Tìm thấy {len(pdf_files)} file luật trong {RAW_LUAT_DIR}")

    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "files": {}
    }

    for pdf_path in tqdm(pdf_files, desc="Xử lý file Luật", unit="file"):
        # --- Kiểm tra blacklist ---
        if pdf_path.name in EXCLUDED_FILES:
            logger.info(f"  [SKIP] {pdf_path.name} — nằm trong danh sách loại trừ")
            stats["skipped"] += 1
            continue

        strategy = FILE_STRATEGIES.get(pdf_path.name)
        logger.info(f"  -> Đang xử lý: {pdf_path.name} | strategy: {strategy}")

        full_content = []
        page_count   = 0
        table_count  = 0
        char_count   = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)

                for page in tqdm(pdf.pages, desc=f"  Trang {pdf_path.stem}", leave=False):
                    # --- FIX DUPLICATE: Lấy bboxes bảng trước ---
                    table_bboxes = get_table_bboxes(page)

                    # --- Trích text + bảng theo ĐÚNG thứ tự vị trí trên trang ---
                    # (thay vì trích toàn bộ text trước rồi nối toàn bộ bảng
                    #  vào cuối như code cũ — xem build_page_blocks() để biết
                    #  vì sao code cũ luôn đẩy bảng xuống cuối trang)
                    page_blocks = build_page_blocks(page, table_bboxes, strategy)

                    for block in page_blocks:
                        if block["type"] == "text":
                            cleaned = clean_text(block["content"], block["strategy"])
                            if cleaned:
                                full_content.append(cleaned)
                                char_count += len(cleaned)
                        else:  # block["type"] == "table"
                            full_content.append("\n\n" + block["content"] + "\n\n")
                            table_count += 1

            # --- Lưu kết quả ---
            output_filename = pdf_path.stem + ".md"
            output_path     = CLEANED_LUAT_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_content))

            stats["processed"] += 1
            stats["files"][pdf_path.name] = {
                "pages": page_count,
                "tables": table_count,
                "chars": char_count,
                "output": str(output_path)
            }
            logger.info(
                f"   ✓ Hoàn tất: {output_filename} "
                f"| {page_count} trang | {table_count} bảng | {char_count:,} ký tự"
            )

        except Exception as e:
            logger.error(f"   ✗ Lỗi khi xử lý {pdf_path.name}: {e}", exc_info=True)
            stats["errors"] += 1

    # --- In thống kê cuối ---
    logger.info("\n" + "="*60)
    logger.info(f"THỐNG KÊ XỬ LÝ LUẬT:")
    logger.info(f"  Tổng:        {stats['total']}")
    logger.info(f"  Thành công:  {stats['processed']}")
    logger.info(f"  Bỏ qua:      {stats['skipped']}")
    logger.info(f"  Lỗi:         {stats['errors']}")

    # Lưu stats ra JSON
    stats_path = CLEANED_LUAT_DIR / "_processing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"  Stats đã lưu tại: {stats_path}")

    return stats


if __name__ == "__main__":
    process_law_pdfs()
