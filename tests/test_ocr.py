import unittest
import pandas as pd
from lwm import config as C, tools as T, ocr_review as OCR


class OCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = T.load_cache()
        cls.decisions = OCR.load_decisions(C.OCR_DECISIONS)
        cls.rules = OCR.effective_rules(T.read_ocr_rules(), cls.decisions)
        cls.review = T.neighbor_review(cls.cache, cls.rules)

    def test_migrated_decisions(self):
        self.assertEqual(self.decisions['decision'].value_counts().to_dict(),
                         {'keep': 1655, 'exclude': 1950, 'review': 33})
        self.assertEqual(len(T.read_ocr_rules()), 164)
        self.assertEqual(len(self.rules), 2114)

    def test_conflicts(self):
        OCR.validate_no_conflicts(self.review, self.decisions)
        bad_review = pd.DataFrame({'neighbor': ['comorn'], 'excluded': [True]})
        bad_decisions = pd.DataFrame({'neighbor': ['comorn'], 'decision': ['keep']})
        with self.assertRaises(ValueError):
            OCR.validate_no_conflicts(bad_review, bad_decisions)

    def test_article_summary(self):
        selected, coverage = OCR.article_neighbors(self.review, self.cache, self.decisions,
            C.IMPERIAL_LEXICON_TARGETS, C.MIN_COUNT_INTERPRET, C.REPORT_TOPN)
        summary = OCR.article_summary(coverage).iloc[0]
        self.assertEqual((summary['pairs'], summary['viewed'], summary['excluded'],
                          summary['retained'], summary['half_or_more'], summary['pending']),
                         (75, 4788, 2538, 2250, 23, 0))
        self.assertEqual(set(selected['target']), set(C.IMPERIAL_LEXICON_TARGETS))
        self.assertTrue(selected['decision'].eq('keep').all())
        self.assertTrue(coverage['count'].ge(C.MIN_COUNT_INTERPRET).all())
        self.assertEqual(coverage.loc[(coverage['period']=='1900s') &
                                      (coverage['target']=='imperial'), 'viewed'].item(), 244)
        for _, group in selected.groupby(['period', 'target']):
            self.assertEqual(group['filtered_rank'].tolist(), list(range(1,31)))
            self.assertTrue(group['raw_rank'].is_monotonic_increasing)

    def test_optional_dictionary(self):
        diagnostic = OCR.diagnose(self.review, self.decisions, None, suggest=False)
        self.assertTrue(diagnostic['spellchecker_unknown'].isna().all())
        self.assertTrue(diagnostic['suggested_correction'].eq('').all())
        self.assertFalse(diagnostic['keep_rule_conflict'].any())

    def test_dictionary_is_diagnostic_only(self):
        class MockSpell:
            def unknown(self, words):
                return {'exampleocr'} & set(words)
            def correction(self, word):
                return 'example'
        review = pd.DataFrame([{'neighbor':'exampleocr', 'period':'1900s',
                                'target':'empire', 'raw_rank':1, 'filtered_rank':1,
                                'cosine':0.7, 'excluded':False}])
        dec = pd.DataFrame(columns=OCR.DECISION_COLUMNS)
        d = OCR.diagnose(review, dec, MockSpell(), suggest=True)
        self.assertEqual(d.loc[0, 'priority'], 'unknown_top30')
        self.assertEqual(d.loc[0, 'suggested_correction'], 'example')
        self.assertFalse(review.loc[0, 'excluded'])

    def test_published_neighbors(self):
        reference = pd.read_csv(C.TAB_DIR/'03_neighbors_review.csv', keep_default_na=False)
        # Missing filtered ranks are numeric NaNs; pd.to_numeric below
        # already handles them. Do not fill numeric columns with strings.
        computed = self.review
        # CSV mixes empty strings and numeric ranks; compare textual fields and ranks separately.
        for col in ['period','target','raw_rank','neighbor','excluded','reason']:
            self.assertEqual(reference[col].tolist(), computed[col].tolist())
        pd.testing.assert_series_equal(pd.to_numeric(reference['filtered_rank'], errors='coerce'),
                                      pd.to_numeric(computed['filtered_rank'], errors='coerce'))
        import numpy as np
        np.testing.assert_allclose(reference['cosine'], computed['cosine'], rtol=0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
