import re
import hashlib
import logging
import yaml
from pathlib import Path
import datetime
from .clean_luat import get_base_dir
from .utilities import setup_logging, count_tokens, refine_content, apply_legal_styling

logger = setup_logging()


# ---------------------------------------------------------------------------
# Form noise: các keyword chỉ ra biểu mẫu nội bộ (không hữu ích cho RAG)
# ---------------------------------------------------------------------------
FORM_NOISE_KEYWORDS = [
    'sổ theo dõi', 'biên bản phân công', 'báo cáo định kỳ',
    'sổ giao nhận', 'nhật ký', 'sổ đăng ký',
    'báo cáo thống kê', 'phiếu xuất kho',
]


# ---------------------------------------------------------------------------
# Regex phát hiện cấu trúc pháp luật
# ---------------------------------------------------------------------------
# Điều: "Điều 5." / "Điều 5:" / "## Điều 5."
RE_DIEU   = re.compile(r"(?:##\s*)?Điều\s+(\d+)[.:]\s*(.*)", re.IGNORECASE)

# Khoản: dòng bắt đầu bằng số + dấu chấm + chữ hoa
# ví dụ: "1. Người điều khiển..."
# Chỉ capture 1 group (số khoản) để split chính xác
RE_KHOAN  = re.compile(r"^(\d+)\.(?=\s+[A-ZĐÀÁẠẢÃẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ])", re.MULTILINE)

# Điểm: "a) ..." / "đ) ..." (chữ thường latin + tiếng Việt 'đ' + ngoặc đơn)
# v2 (2026-05-19): bổ sung 'đ' và cho phép Điểm cùng dòng sau dấu ';' — vì
# format pháp lý VN thường có "d) ...; đ) ..." cùng dòng (xem TT 17/2018/VPCP).
RE_DIEM   = re.compile(r"(?:^|;\s+)([a-zđ])\)(?=\s+)", re.MULTILINE)

# Ngưỡng split (token-level)
THRESH_DIEU          = 600   # Điều > ngưỡng này → split xuống Khoản
THRESH_KHOAN_DEFAULT = 500   # Khoản > ngưỡng → split xuống Điểm (Luật/TT)
THRESH_KHOAN_STRICT  = 80    # Strict cho Nghị định phạt (cần granular Điểm)

# MIN_CHUNK tách theo level (v2): L1/L2 giữ 30 (tránh noise); L3 hạ 5 vì
# mỗi Điểm pháp lý vốn ngắn (1 hành vi = 1 câu 10-20 token), nhưng vẫn là
# target retrieval độc lập. Filter 30 ở Điểm gây MẤT ~70% Điểm trong NĐ 168.
MIN_CHUNK         = 30   # giữ tương thích — code cũ vẫn dùng được
MIN_CHUNK_DIEM    = 5    # L3 (Điểm) — hạ xuống 5 token


class HierarchicalLegalSplitter:
    """
    Tách văn bản pháp luật theo 3 tầng phân cấp:
      Level 1 — Điều   (≤ 600 token)
      Level 2 — Khoản  (nếu Điều > 600 token)
      Level 3 — Điểm   (nếu Khoản > 500 token)
      Fallback         (văn bản xuôi, không có cấu trúc Điều/Khoản)
    """

    def split_into_articles(self, full_text: str) -> list[tuple[str, str, str]]:
        """
        Tách full text thành list of (dieu_num, dieu_title, dieu_content).
        """
        articles = []
        lines    = full_text.splitlines()

        current_num     = "0"
        current_title   = "Phần giới thiệu"
        current_lines   = []

        for line in lines:
            m = RE_DIEU.match(line.strip())
            if m:
                # Lưu Điều hiện tại
                if current_lines:
                    articles.append((current_num, current_title, "\n".join(current_lines)))
                current_num   = m.group(1)
                current_title = m.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Lưu Điều cuối
        if current_lines:
            articles.append((current_num, current_title, "\n".join(current_lines)))

        return articles

    def split_article_into_khoans(self, text: str) -> list[tuple[str, str]]:
        """
        Tách nội dung một Điều thành list of (khoan_num, khoan_content).
        """
        parts = RE_KHOAN.split(text)
        khoans = []

        if len(parts) <= 1:
            # Không có Khoản rõ ràng — cả Điều là 1 block
            return [("0", text)]

        # parts[0] = text trước Khoản 1 (thường là tiêu đề Điều)
        if parts[0].strip():
            khoans.append(("0", parts[0].strip()))

        # Mỗi Khoản: parts[i] = khoan_num, parts[i+1] = content_between
        i = 1
        while i + 1 < len(parts):
            khoan_num = parts[i]
            content   = parts[i + 1]
            khoans.append((khoan_num, f"{khoan_num}.{content}"))
            i += 2

        return khoans if khoans else [("0", text)]

    def split_khoan_into_diems(self, text: str, enrich_preamble: bool = True) -> list[tuple[str, str]]:
        """
        Tách nội dung một Khoản thành list of (diem_label, diem_content).

        v2 (2026-05-19): nếu `enrich_preamble=True`, mỗi L3 chunk được
        PREPEND phần preamble của Khoản (cụm "Phạt tiền X-Y đồng đối với ..."
        trước Điểm đầu tiên). Lý do: Điểm pháp lý thường chỉ chứa mô tả hành
        vi (~10-20 token), không có số tiền — cần preamble để chunk tự chứa
        đủ thông tin {mức phạt + hành vi} cho retrieval Điểm-level chính xác.
        """
        parts = RE_DIEM.split(text)
        diems = []

        if len(parts) <= 1:
            return [("", text)]

        preamble = parts[0].strip() if parts[0].strip() else ""
        if preamble:
            diems.append(("", preamble))

        i = 1
        while i + 1 < len(parts):
            label   = parts[i]
            content = parts[i + 1]
            diem_body = f"{label}){content}".strip()
            if enrich_preamble and preamble:
                # Prepend preamble so each Điểm chunk is self-contained:
                # "Phạt tiền X-Y đồng đối với...\nk) Dàn hàng ngang từ 03 xe..."
                enriched = f"{preamble}\n{diem_body}"
            else:
                enriched = diem_body
            diems.append((label, enriched))
            i += 2

        return diems if diems else [("", text)]

    def chunk_document(
        self,
        full_text:   str,
        doc_meta:    dict,
        doc_type:    str,
        source_file: str,
    ) -> list[dict]:
        """
        Entry point chính: nhận full text → trả về list chunk dicts.
        """
        articles = self.split_into_articles(full_text)
        chunks   = []
        seen_hashes = set()  # Deduplication

        for dieu_num, dieu_title, dieu_content in articles:
            if count_tokens(dieu_content) < MIN_CHUNK:
                continue  # Quá ngắn — bỏ qua

            if count_tokens(dieu_content) <= THRESH_DIEU:
                # ── Level 1: Cả Điều là 1 chunk ──
                chunk = self._make_chunk(
                    text        = dieu_content,
                    doc_meta    = doc_meta,
                    doc_type    = doc_type,
                    source_file = source_file,
                    dieu_num    = dieu_num,
                    dieu_title  = dieu_title,
                    khoan_num   = None,
                    diem_label  = None,
                    level       = 1,
                )
                chunks.append(chunk)
            else:
                # ── Level 2: Split theo Khoản ──
                # v2 (2026-05-19): Logic split L3 đổi từ pure threshold sang
                # "split nếu nghidinh + có ≥2 Điểm OR Khoản > THRESH". Nghị
                # định phạt cần granular L3 chunks để retrieval Điểm-level
                # chính xác — kể cả Khoản nhỏ (vd Đ7 K10 = 71 token, 4 Điểm).
                khoans = self.split_article_into_khoans(dieu_content)
                # Pick threshold based on document type
                effective_thresh = (
                    THRESH_KHOAN_STRICT if doc_type == "nghidinh"
                    else THRESH_KHOAN_DEFAULT
                )

                for khoan_num, khoan_content in khoans:
                    if count_tokens(khoan_content) < MIN_CHUNK:
                        continue

                    # Decide whether to split into L3 Điểm chunks.
                    # Trigger split if EITHER (a) Khoản is large (size-based)
                    # OR (b) Nghị định with ≥2 distinct Điểm (semantic-based).
                    has_multi_diem = len(RE_DIEM.findall(khoan_content)) >= 2
                    should_split_l3 = (
                        count_tokens(khoan_content) > effective_thresh
                        or (doc_type == "nghidinh" and has_multi_diem)
                    )

                    if not should_split_l3:
                        chunk = self._make_chunk(
                            text        = khoan_content,
                            doc_meta    = doc_meta,
                            doc_type    = doc_type,
                            source_file = source_file,
                            dieu_num    = dieu_num,
                            dieu_title  = dieu_title,
                            khoan_num   = khoan_num if khoan_num != "0" else None,
                            diem_label  = None,
                            level       = 2,
                        )
                        chunks.append(chunk)
                    else:
                        # ── Level 3: Split theo Điểm ──
                        # enrich_preamble=True: mỗi L3 chunk chứa cụm
                        # "Phạt tiền X-Y đồng..." của Khoản preamble + nội dung
                        # Điểm cụ thể → chunk self-contained.
                        diems = self.split_khoan_into_diems(
                            khoan_content, enrich_preamble=True,
                        )
                        for diem_label, diem_content in diems:
                            # v2: MIN_CHUNK_DIEM (=5) thay MIN_CHUNK (=30)
                            # vì Điểm pháp lý ngắn nhưng vẫn là target retrieval.
                            if count_tokens(diem_content) < MIN_CHUNK_DIEM:
                                continue
                            chunk = self._make_chunk(
                                text        = diem_content,
                                doc_meta    = doc_meta,
                                doc_type    = doc_type,
                                source_file = source_file,
                                dieu_num    = dieu_num,
                                dieu_title  = dieu_title,
                                khoan_num   = khoan_num if khoan_num != "0" else None,
                                diem_label  = diem_label if diem_label else None,
                                level       = 3,
                            )
                            chunks.append(chunk)

        # [TASK 3c] Lọc biểu mẫu nhiễu nội bộ + Deduplication
        unique_chunks = []
        form_noise_removed = 0
        for chunk in chunks:
            # Kiểm tra biểu mẫu nhiễu nội bộ
            content_lower = chunk['content'].lower()
            if any(kw in content_lower for kw in FORM_NOISE_KEYWORDS):
                form_noise_removed += 1
                logger.debug(f"Form noise bị loại: {chunk['metadata']['chunk_id']}")
                continue

            # Deduplication dựa trên content hash
            h = hashlib.md5(chunk['content'].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_chunks.append(chunk)
            else:
                logger.debug(f"Duplicate chunk bị loại: {chunk['metadata']['chunk_id']}")

        if form_noise_removed:
            logger.info(f"  → Lọc {form_noise_removed} chunk biểu mẫu nhiễu nội bộ")

        return unique_chunks

    def _make_chunk(
        self,
        text:        str,
        doc_meta:    dict,
        doc_type:    str,
        source_file: str,
        dieu_num:    str,
        dieu_title:  str,
        khoan_num:   str | None,
        diem_label:  str | None,
        level:       int,
    ) -> dict:
        """Tạo chunk dict với metadata đầy đủ + refinement + context enrichment."""
        # Tạo chunk_id chuẩn hóa
        doc_slug   = doc_meta.get("doc_id", source_file).replace("/", "_").replace("-", "_")
        khoan_part = f"_khoan{khoan_num}" if khoan_num else ""
        diem_part  = f"_diem{diem_label}" if diem_label else ""
        chunk_id   = f"{doc_slug}_dieu{dieu_num}{khoan_part}{diem_part}"

        # ── BƯỚC MỚI v2.1: Regex Refinement — sửa lỗi ngắt dòng PDF ──
        content = refine_content(text)

        # ── [TASK 4] Legal Styling: Bold mức phạt và hành vi vi phạm ──
        content = apply_legal_styling(content)

        # ── [TASK 2] Context Enrichment ──
        ten_van_ban = doc_meta.get("title", source_file)
        if dieu_num and dieu_num != "0":
            context_line = f"Văn bản: {ten_van_ban} | Điều {dieu_num}: {dieu_title}"
            if not content.startswith(context_line):
                content = f"{context_line}\n{content}"

        # Chèn warning vào đầu mỗi chunk nếu văn bản có cảnh báo
        if doc_meta.get("warning") and not content.startswith(">"):
            content = f"> ⚠️ {doc_meta['warning']}\n\n{content}"

        return {
            "metadata": {
                # Trường bắt buộc theo prompt
                "chunk_id":       chunk_id,
                "doc_id":         doc_meta.get("doc_id", source_file),
                "ten_van_ban":    doc_meta.get("title", source_file),
                "issuer":         doc_meta.get("issuer", ""),
                "document_type":  doc_type,
                "status":         doc_meta.get("status", "active"),
                "effective_date": doc_meta.get("effective_date", ""),
                "topic":          doc_meta.get("topic", "Chung"),

                # Trường pháp lý phân cấp
                "dieu":       int(dieu_num) if dieu_num.isdigit() else dieu_num,
                "dieu_title": dieu_title,
                "khoan":      khoan_num,
                "diem":       diem_label,
                "level":      level,

                # Trường kỹ thuật
                "source_file":   source_file,
                "token_estimate": count_tokens(content),
            },
            "content": content,
        }