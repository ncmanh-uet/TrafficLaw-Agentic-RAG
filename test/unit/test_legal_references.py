"""Offline scope checks for legal-reference links used by Faithfulness."""

import unittest

from test.evaluation_agent.legal_references import (
    LegalLocation, parse_references, reference_relation, refers_to,
)


class LegalReferenceTests(unittest.TestCase):
    DOC = "999/2099/NĐ-CP"

    def location(self, paragraph=None, point=None, *, article="7", doc=None):
        return LegalLocation(doc or self.DOC, article, paragraph, point)

    def parse(self, text):
        return parse_references(text, doc_id=self.DOC, article="7")

    def test_semicolon_point_lists_do_not_borrow_letters(self):
        text = ("b) Thực hiện hành vi quy định tại điểm đ khoản 4; điểm a khoản 6; "
                "điểm c, điểm d, điểm đ khoản\n7; điểm a khoản 8 Điều này "
                "bị trừ 04 điểm.")
        self.assertEqual(self.parse(text), [self.location("4", "đ"), self.location("6", "a"),
                         self.location("7", "c"), self.location("7", "d"),
                         self.location("7", "đ"), self.location("8", "a")])
        self.assertTrue(refers_to(text, self.location("13", "b"), self.location("7", "c")))
        self.assertFalse(refers_to(text, self.location("13", "b"), self.location("8", "b")))

    def test_point_then_whole_paragraph_does_not_inherit_point(self):
        text = "d) Thực hiện hành vi quy định tại điểm b khoản 8, khoản 10 Điều này bị trừ 10 điểm."
        self.assertEqual(self.parse(text), [self.location("8", "b"), self.location("10")])
        self.assertTrue(refers_to(text, self.location("13", "d"), self.location("10", "a")))
        self.assertFalse(refers_to(text, self.location("13", "d"), self.location("8", "a")))

    def test_four_points_then_whole_paragraph(self):
        text = "điểm d, điểm đ, điểm e, điểm g khoản 9; khoản 11 Điều này"
        self.assertEqual(self.parse(text), [self.location("9", point) for point in ("d", "đ", "e", "g")]
                         + [self.location("11")])

    def test_abbreviated_point_list(self):
        self.assertEqual(self.parse("các điểm c, d và đ khoản 7 Điều này"),
                         [self.location("7", point) for point in ("c", "d", "đ")])

    def test_paragraph_list(self):
        self.assertEqual(self.parse("các khoản 1, 2 và 3 Điều 9"),
                         [self.location(value, article="9") for value in ("1", "2", "3")])

    def test_explicit_article_scopes_complete_preceding_list(self):
        text = "điểm a khoản 2; điểm b khoản 3 Điều 9"
        self.assertEqual(self.parse(text), [self.location("2", "a", article="9"),
                                           self.location("3", "b", article="9")])

    def test_multiple_explicit_articles_have_separate_scope(self):
        text = "điểm a khoản 2 Điều 9; điểm b khoản 3 Điều 10"
        self.assertEqual(self.parse(text), [self.location("2", "a", article="9"),
                                           self.location("3", "b", article="10")])

    def test_same_article_in_another_document_is_not_interchangeable(self):
        text = "điểm a khoản 2 Điều này"
        self.assertFalse(refers_to(text, self.location("13", "b"),
                                   self.location("2", "a", doc="998/2099/NĐ-CP")))

    def test_explicit_foreign_numbered_document_is_resolved(self):
        text = "điểm a khoản 2 Điều 9 của Nghị định số 998/2099/NĐ-CP"
        self.assertEqual(self.parse(text), [self.location("2", "a", article="9", doc="998/2099/NĐ-CP")])
        self.assertTrue(refers_to(text, self.location("13", "b"),
                                  self.location("2", "a", article="9", doc="998/2099/NĐ-CP")))
        self.assertFalse(refers_to(text, self.location("13", "b"), self.location("2", "a", article="9")))

    def test_unnumbered_foreign_name_is_not_assumed_to_be_source_document(self):
        self.assertEqual(self.parse("điểm a khoản 2 Điều 9 của Luật Giả định"), [])
        self.assertEqual(self.parse("điểm a khoản 2 Điều 9 của Luật Giả định thay thế Luật số 998/2099/QH99"), [])

    def test_explicit_this_document(self):
        self.assertEqual(self.parse("điểm a khoản 2 Điều 9 của Nghị định này"),
                         [self.location("2", "a", article="9")])

    def test_document_type_part_of_identity(self):
        self.assertFalse(refers_to("điểm a khoản 2 Điều này", self.location("13", "b"),
                                   self.location("2", "a", doc="999/2099/TT-BCA")))

    def test_wrong_article_and_clause_do_not_match(self):
        text = "điểm a khoản 2 Điều 9"
        source = self.location("13", "b")
        self.assertFalse(refers_to(text, source, self.location("2", "a")))
        self.assertFalse(refers_to(text, source, self.location("3", "a", article="9")))

    def test_point_label_is_not_a_reference(self):
        self.assertEqual(self.parse("b) Điều khiển phương tiện thuộc diện xử phạt."), [])
        self.assertFalse(refers_to("b) Hành vi tại điểm a khoản 2 Điều này", self.location("13", "b"),
                                   self.location("2", "b")))

    def test_heading_is_not_whole_article_reference(self):
        self.assertEqual(self.parse("Điều 7. Xử phạt hành vi\nđiểm a khoản 2 Điều này"),
                         [self.location("2", "a")])

    def test_sentence_final_article_period_is_not_a_heading(self):
        self.assertEqual(self.parse("Hành vi tại điểm a khoản 2 Điều 9."),
                         [self.location("2", "a", article="9")])
        self.assertEqual(self.parse("Được quy định tại Điều 9."), [self.location(article="9")])

    def test_source_metadata_headers_are_not_broad_cross_references(self):
        text = ("[SOURCE_ID=S15] [INTERNAL:SIBLING] Nghị định 999/2099/NĐ-CP · Điều 7 · Khoản 13 · Điểm b\n"
                "Văn bản: Nghị định 999/2099/NĐ-CP | Điều 7: Nội dung điều\n"
                "13. Trừ điểm:\n"
                "b) Hành vi quy định tại điểm a khoản 8 Điều này bị trừ 04 điểm.")
        self.assertEqual(self.parse(text), [self.location("8", "a")])
        self.assertFalse(refers_to(text, self.location("13", "b"), self.location("8", "b")))
        self.assertIs(reference_relation(text, self.location("13", "b"), self.location("8", "b")), False)

    def test_d_and_crossed_d_are_distinct_and_nfc_equivalent(self):
        source = self.location("13", "b")
        self.assertTrue(refers_to("ĐIỂM Đ KHOẢN 2 ĐIỀU NÀY", source, self.location("2", "đ")))
        self.assertFalse(refers_to("điểm đ khoản 2 Điều này", source, self.location("2", "d")))
        import unicodedata
        self.assertEqual(self.parse(unicodedata.normalize("NFD", "điểm đ khoản 2 Điều này")),
                         [self.location("2", "đ")])

    def test_unknown_forms_do_not_widen_point_to_paragraph(self):
        for text in ("điểm a của khoản 2 Điều này", "điểm a đến điểm c khoản 2 Điều này",
                     "điểm a khoản này", "điểm a, b Điều này"):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text), [])

    def test_point_reference_cannot_establish_entire_paragraph(self):
        self.assertFalse(refers_to("điểm a khoản 2 Điều này", self.location("13", "b"), self.location("2")))
        self.assertTrue(refers_to("khoản 2 Điều này", self.location("13", "b"), self.location("2", "a")))

    def test_invalid_or_missing_locations_are_not_links(self):
        for location in (LegalLocation("", "7", "2", "a"), LegalLocation(self.DOC, "?", "2", "a"),
                         LegalLocation(self.DOC, "7", None, "a"), LegalLocation(self.DOC, "7", "bad", "a")):
            with self.subTest(location=location):
                self.assertFalse(refers_to("điểm a khoản 2 Điều này", self.location("13", "b"), location))

    def test_duplicates_are_removed_and_numeric_ids_are_normalized(self):
        self.assertEqual(self.parse("điểm a khoản 02; điểm a khoản 2 Điều này"), [self.location("2", "a")])

    def test_comma_before_paragraph_does_not_drop_the_point_group(self):
        text = ("b) Vi phạm quy định tại điểm a khoản 1; điểm b, điểm g, điểm h, "
                "điểm n, điểm o, điểm p, khoản 3; điểm a khoản 4 Điều này mà gây tai nạn.")
        expected = [self.location("1", "a")]
        expected += [self.location("3", point) for point in ("b", "g", "h", "n", "o", "p")]
        expected += [self.location("4", "a")]
        self.assertEqual(self.parse(text), expected)
        self.assertIs(reference_relation(text, self.location("10", "b"), self.location("3", "b")), True)
        self.assertIs(reference_relation(text, self.location("10", "b"), self.location("3", "c")), False)

    def test_enumerated_list_introduction_is_not_whole_article_reference(self):
        text = ("b) Vi phạm quy định tại một trong các điểm, khoản sau của Điều này "
                "mà gây tai nạn giao thông:\nđiểm a khoản 1; điểm b khoản 3 Điều này.")
        self.assertEqual(self.parse(text), [self.location("1", "a"), self.location("3", "b")])
        self.assertIs(reference_relation(text, self.location("10", "b"), self.location("3", "b")), True)
        self.assertIs(reference_relation(text, self.location("10", "b"), self.location("3", "c")), False)

    def test_complete_reference_list_can_certify_negative_relation(self):
        text = "điểm a khoản 6; điểm c, điểm d, điểm đ khoản 7; điểm a khoản 8 Điều này"
        self.assertIs(reference_relation(text, self.location("13", "b"), self.location("8", "b")), False)
        self.assertIs(reference_relation(text, self.location("13", "b"), self.location("7", "c")), True)

    def test_ordinary_penalty_word_does_not_mask_later_legal_reference(self):
        text = ("g) Ngoài việc bị trừ điểm, thực hiện hành vi quy định tại "
                "điểm a khoản 6 Điều này bị phạt thêm 20 đơn vị.")
        self.assertEqual(self.parse(text), [self.location("6", "a")])
        self.assertIs(reference_relation(text, self.location("13", "g"), self.location("6", "a")), True)
        self.assertIs(reference_relation(text, self.location("13", "g"), self.location("6", "b")), False)

    def test_incomplete_mixed_list_cannot_certify_negative_relation(self):
        for unknown in ("điểm a khoản này", "điểm a đến điểm c khoản 3", "khoản 3 đến khoản 5",
                        "điểm b của khoản 3", "điểm a, b Điều này",
                        "điểm b khoản 3 Điều 9 của Luật Giả định"):
            text = "điểm a khoản 1 Điều này; " + unknown
            with self.subTest(unknown=unknown):
                self.assertIsNone(reference_relation(text, self.location("13", "b"), self.location("3", "b")))
                # A separate, fully addressed known link remains usable.
                self.assertIs(reference_relation(text, self.location("13", "b"), self.location("1", "a")), True)

    def test_absent_or_unknown_reference_has_no_certified_negative(self):
        for text in ("b) Hành vi không có dẫn chiếu.", "điểm a khoản này",
                     "khoản 3 Điều 9 của Luật Giả định"):
            with self.subTest(text=text):
                self.assertIsNone(reference_relation(text, self.location("13", "b"), self.location("3", "b")))

    def test_range_is_unknown_for_unlisted_intermediate_locations(self):
        text = "khoản 1 Điều này; khoản 3 đến khoản 5 Điều này"
        self.assertIsNone(reference_relation(text, self.location("13", "b"), self.location("4")))


if __name__ == "__main__":
    unittest.main()
