# Một số hàm tiện ích cho chunker

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


BASE_DIR = get_base_dir()
LOG_DIR     = BASE_DIR / "logs"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"chunker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (word-based, đủ cho ngưỡng split)
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Ước tính số token bằng cách đếm từ (1 từ ≈ 1.3 token cho tiếng Việt)."""
    words = len(text.split())
    return int(words * 1.3)


def refine_content(text: str) -> str:
    """
    Sửa các lỗi phổ biến khi extract PDF:
      1. Từ bị ngắt bởi dấu gạch nối cuối dòng + xuống dòng
      2. Xuống dòng vô nghĩa giữa câu
      3. Dòng kẻ trang trí, dòng trống liên tiếp
    """
    # ── Bảo vệ bảng Markdown: tạm thay thế các dòng bảng bằng placeholder ──
    table_lines = []
    lines = text.split('\n')
    protected = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('|') or (re.match(r'^[-:|]+$', stripped) and '|' in stripped):
            table_lines.append((i, line))
            protected.append(f'__TABLE_LINE_{i}__')
        else:
            protected.append(line)
    text = '\n'.join(protected)

    # [TASK 1a] Nối từ bị ngắt: giữ dấu gạch nối, chỉ xóa xuống dòng vô nghĩa
    text = re.sub(r'(\s*-\s*)\n\s*', r'\1', text)

    # [TASK 1b] Nối dòng bị ngắt giữa câu
    text = re.sub(
        r'(?<=[a-zàáạảãắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ,;])'
        r'\n'
        r'(?=[a-zàáạảãắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ])',
        ' ', text
    )

    # [TASK 1c] Xóa dòng kẻ trang trí (5+ ký tự ---/===/___ liên tiếp)
    text = re.sub(r'^[-=_]{5,}\s*$', '', text, flags=re.MULTILINE)

    # [TASK 1d] Gộp dòng trống liên tiếp (>2 dòng → 1 dòng trống)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── Khôi phục dòng bảng Markdown ──
    lines_out = text.split('\n')
    for idx, original_line in table_lines:
        for j, line in enumerate(lines_out):
            if line.strip() == f'__TABLE_LINE_{idx}__':
                lines_out[j] = original_line
                break

    return '\n'.join(lines_out).strip()

# ---------------------------------------------------------------------------
# Legal Styling: Bold các mức phạt và hành vi vi phạm quan trọng
# ---------------------------------------------------------------------------
def apply_legal_styling(text: str) -> str:
    """Bold các mức phạt tiền và hình thức xử phạt bổ sung quan trọng."""
    # Bold mức phạt tiền:  "phạt tiền từ X đồng đến Y đồng"
    text = re.sub(
        r'(phạt tiền từ\s+[\d.,]+\s+đồng\s+đến\s+[\d.,]+\s+đồng)',
        r'**\1**', text, flags=re.IGNORECASE
    )
    # Bold tước quyền sử dụng GPLX
    text = re.sub(
        r'(tước quyền sử dụng[^.;]{5,80}tháng)',
        r'**\1**', text, flags=re.IGNORECASE
    )
    # Tránh bold lồng (** bên trong **)
    text = re.sub(r'\*{4,}', '**', text)
    return text



# ---------------------------------------------------------------------------
# Lưu chunk ra file .md + YAML frontmatter
# ---------------------------------------------------------------------------
def save_chunk_as_md(chunk: dict, output_dir: Path):
    """Lưu 1 chunk ra file Markdown với YAML frontmatter."""
    meta     = chunk["metadata"]
    chunk_id = meta["chunk_id"]

    # Tên file từ chunk_id (giới hạn độ dài để tránh OSError: File name too long)
    # Hầu hết OS giới hạn 255 chars, ta để 150 cho an toàn
    safe_name = re.sub(r"[^\w\-]", "_", chunk_id)
    if len(safe_name) > 150:
        # Nếu quá dài, lấy 140 ký tự đầu + 8 ký tự hash của phần còn lại
        suffix = hashlib.md5(safe_name.encode()).hexdigest()[:8]
        safe_name = safe_name[:140] + "_" + suffix
    
    out_path = output_dir / f"{safe_name}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(meta, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        f.write("---\n\n")
        f.write(chunk["content"])