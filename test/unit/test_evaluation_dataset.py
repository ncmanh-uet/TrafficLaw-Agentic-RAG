"""Offline integrity and source-anchor checks for the local agent gold dataset.

These checks do not replace semantic review and do not invoke an LLM.
Data/ is intentionally not distributed in every checkout.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest

from test.evaluation_agent.scoring_data import dataset_rows, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data/test"


@unittest.skipUnless(all((DATA / name).exists() for name in (
    "evaluation2.jsonl", "evaluation2_support.jsonl", "evaluation2.manifest.json", "evaluation.jsonl")),
    "Local benchmark artifacts are not present")
class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = dataset_rows(DATA / "evaluation2.jsonl")
        cls.by_id = {r["query_id"]: r for r in cls.rows}
        cls.support = read_jsonl(DATA / "evaluation2_support.jsonl")
        cls.manifest = json.loads((DATA / "evaluation2.manifest.json").read_text(encoding="utf-8"))

    def test_five_fields_counts_and_manifest_splits(self):
        self.assertEqual(len(self.rows), 65)
        self.assertEqual(len({r["query"] for r in self.rows}), 65)
        fields = {"query_id", "query", "reference_answer", "category", "split"}
        for row in self.rows:
            self.assertEqual(set(row), fields)
        self.assertEqual(Counter(r["category"] for r in self.rows),
                         {"internal_answerable": 50, "web_required": 10, "insufficient_evidence": 5})
        self.assertEqual(Counter(r["split"] for r in self.rows), {"pilot": 10, "test": 55})
        for split, ids in self.manifest["splits"].items():
            self.assertEqual(sorted(ids), sorted(r["query_id"] for r in self.rows if r["split"] == split))

    def test_file_hashes_and_unchanged_retriever_dataset(self):
        for filename, expected in (
            ("evaluation2.jsonl", self.manifest["sha256"]),
            ("evaluation2_support.jsonl", self.manifest["support_sha256"]),
            ("evaluation.jsonl", self.manifest["provenance"]["original_dataset_sha256"]),
        ):
            self.assertEqual(hashlib.sha256((DATA / filename).read_bytes()).hexdigest(), expected)

    def test_sources_hashes_quotes_and_web_spans_resolve(self):
        self.assertEqual(len(self.support), 65)
        self.assertEqual({r["query_id"] for r in self.support}, set(self.by_id))
        self.assertEqual(len({r["query_id"] for r in self.support}), 65)
        for row in self.support:
            with self.subTest(query_id=row["query_id"]):
                sources = {s["source_id"]: s for s in row["reference_sources"]}
                self.assertEqual(len(sources), len(row["reference_sources"]))
                for source in sources.values():
                    self.assertEqual(hashlib.sha256(source["content"].encode("utf-8")).hexdigest(),
                                     source["content_sha256"])
                for fact in row["required_facts"]:
                    self.assertTrue(fact["text"].strip())
                    self.assertTrue(fact["supporting_evidence"])
                    for evidence in fact["supporting_evidence"]:
                        content = sources[evidence["source_id"]]["content"]
                        if "quote" in evidence:
                            self.assertIn(evidence["quote"], content)
                        else:
                            start, end = evidence["span_start"], evidence["span_end"]
                            self.assertTrue(0 <= start < end <= len(content))
                            self.assertTrue(content[start:end].strip())

    def test_scenario_groups_and_refusal_policies_unchanged(self):
        groups = defaultdict(set)
        for row in self.support:
            gold = self.by_id[row["query_id"]]
            groups[row["leakage_group"]].add(gold["split"])
            if gold["category"] == "insufficient_evidence":
                self.assertEqual(row["metric_policy"]["factual_correctness"], "manual_abstention_review_not_factual_reference")
                self.assertTrue(row["abstention_rubric"]["required_behaviors"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_critical_references_keep_explicit_context_and_source_conditions(self):
        required = {
            "q_004": ("Điều 69 Luật Đường bộ 2024", "xe ô tô cứu thương", "dấu hiệu nhận diện",
                      "niêm yết thông tin", "giám sát hành trình", "hình ảnh người lái xe"),
            "q_006": ("đường thôn", "ngõ, ngách, kiệt, hẻm", "công trình khác"),
            "q_009": ("mục đích an ninh", "Cục Cảnh sát giao thông"),
            "q_017": ("giá vé niêm yết", "hàng có mùi hôi thối"),
            "q_021": ("miễn trừ ngoại giao", "điều ước quốc tế"),
            "q_036": ("90 ngày", "công chứng tại Việt Nam", "lỗi kỹ thuật", "Miễn giấy khám sức khỏe"),
            "q_040": ("Điều 7 Thông tư 35/2024/TT-BGTVT", "136", "67", "203", "152", "83", "235",
                      "1.000", "1.100"),
            "q_049": ("10 ngày", "Trưởng Công an cấp xã", "chưa có mạng kết nối", "Công an cấp huyện",
                      "số điện thoại liên hệ"),
            "q_050": ("tiếp nhận biên bản", "đối chiếu", "mất biên bản", "hình thức, mức xử phạt",
                      "thông báo trừ điểm"),
            "q_052": ("30/4/2025", "tỉnh liền kề", "nhận vé", "không phải trả tiền vé"),
        }
        for query_id, fragments in required.items():
            with self.subTest(query_id=query_id):
                reference = self.by_id[query_id]["reference_answer"]
                for text in fragments:
                    self.assertIn(text.casefold(), reference.casefold())

    def test_references_do_not_require_unasked_details_or_evaluator_instructions(self):
        self.assertNotIn("nội tỉnh", self.by_id["q_052"]["reference_answer"])
        self.assertNotIn("Đơn vị vận tải", self.by_id["q_052"]["reference_answer"])
        self.assertNotIn("16 xe", self.by_id["q_057"]["reference_answer"])
        self.assertNotIn("tạm giữ giấy tờ", self.by_id["q_043"]["reference_answer"])
        self.assertNotIn("Không tự suy ra", self.by_id["q_042"]["reference_answer"])
        self.assertNotIn("Thông tin được hỏi", self.by_id["q_053"]["reference_answer"])
        # Scope review v6: facts that exist in the source do not automatically
        # become required answers when the question asks something narrower.
        self.assertNotIn("lệ phí đăng ký", self.by_id["q_008"]["reference_answer"])
        self.assertNotIn("nội dung phù hợp", self.by_id["q_016"]["reference_answer"])
        self.assertNotIn("sát hạch viên", self.by_id["q_016"]["reference_answer"])
        self.assertNotIn("đổi biển số", self.by_id["q_022"]["reference_answer"])
        self.assertNotIn("giữ biển", self.by_id["q_023"]["reference_answer"])
        self.assertNotIn("tuyến đường", self.by_id["q_030"]["reference_answer"])
        self.assertNotIn("chụp ảnh", self.by_id["q_038"]["reference_answer"])
        self.assertNotIn("ô tô điện", self.by_id["q_040"]["reference_answer"])
        self.assertNotIn("Mẫu 10", self.by_id["q_042"]["reference_answer"])
        self.assertNotIn("tước trên môi trường điện tử", self.by_id["q_050"]["reference_answer"])
        self.assertNotIn("App VNeCSGT", self.by_id["q_050"]["reference_answer"])
        self.assertNotIn("một tháng", self.by_id["q_055"]["reference_answer"])
        self.assertNotIn("chỉ ô tô", self.by_id["q_057"]["reference_answer"])

    def test_q049_covers_resolution_and_owner_duties_with_exact_new_sources(self):
        reference = self.by_id["q_049"]["reference_answer"]
        for fragment in ("20 ngày", "chưa nhận được", "phải hợp tác", "chiếm đoạt",
                         "cấp xã", "cấp huyện", "Đăng kiểm", "gỡ cảnh báo", "thông báo ngay"):
            self.assertIn(fragment, reference)
        support = next(r for r in self.support if r["query_id"] == "q_049")
        source_ids = {s["source_id"] for s in support["reference_sources"]}
        self.assertTrue({"73_2024_TT_BCA_dieu24_khoan2", "73_2024_TT_BCA_dieu24_khoan3",
                         "73_2024_TT_BCA_dieu24_khoan4", "73_2024_TT_BCA_dieu24_khoan5",
                         "168_2024_NĐ_CP_dieu47_khoan8_diemb"} <= source_ids)

    def test_internal_snapshots_match_corpus_not_just_their_own_hashes(self):
        corpus_path = ROOT / "Data/all_chunks.jsonl"
        if not corpus_path.exists():
            self.skipTest("Local corpus is not present")
        corpus = defaultdict(list)
        for chunk in read_jsonl(corpus_path):
            corpus[chunk["metadata"]["chunk_id"]].append(chunk)
        for row in self.support:
            for source in row["reference_sources"]:
                if source["source_type"] != "internal":
                    continue
                with self.subTest(query_id=row["query_id"], source_id=source["source_id"]):
                    matches = [c for c in corpus[source["source_id"]] if c["content"] == source["content"]]
                    self.assertTrue(matches)
                    self.assertTrue(any(c["metadata"]["doc_id"] == source["doc_id"] and
                                        all(str(c["metadata"].get(k)) == str(source["legal_location"].get(k))
                                            for k in ("dieu", "khoan", "diem")) for c in matches))


if __name__ == "__main__":
    unittest.main()
