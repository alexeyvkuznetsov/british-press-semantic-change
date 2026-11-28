"""Check consistency of exported publication data without loading models or pickle."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results' / 'tables'
PERIODS = [f'{year}-е' for year in range(1800, 1920, 10)]
TARGETS = ['imperialism', 'imperialist', 'imperialists', 'jingoism',
           'empire', 'imperial', 'emperor', 'empress']


class PublicationTables(unittest.TestCase):
    def test_imperial_balance(self):
        freq = pd.read_csv(TABLES / '01_word_frequencies.csv')
        counts = freq.pivot(index='десятилетие', columns='слово', values='count').loc[PERIODS, TARGETS]
        main = pd.read_csv(TABLES / '04_empire_balance.csv', index_col=0).loc[PERIODS, TARGETS]
        alt = pd.read_csv(TABLES / '04_geographic_balance.csv', index_col=0).loc[PERIODS, TARGETS]
        self.assertTrue(main.notna().equals(counts.ge(50)))
        self.assertTrue(alt.notna().equals(main.notna()))
        self.assertEqual(int(main.notna().sum().sum()), 75)
        delta = (main - alt).abs().stack()
        self.assertAlmostEqual(delta.mean(), 0.044281333333, places=6)
        self.assertAlmostEqual(delta.max(), 0.1122, places=4)

    def test_imperial_neighbors(self):
        data = pd.read_csv(TABLES / '03_neighbors_article1_top30.csv')
        self.assertEqual(len(data), 2250)
        self.assertEqual(data.groupby(['period', 'target']).ngroups, 75)
        for _, group in data.groupby(['period', 'target']):
            self.assertEqual(group['filtered_rank'].tolist(), list(range(1, 31)))
            self.assertTrue(group['raw_rank'].is_monotonic_increasing)
            self.assertTrue(group['decision'].eq('keep').all())

    def test_general_frequencies(self):
        words = ['empire', 'imperial', 'emperor', 'empress']
        freq = pd.read_csv(TABLES / '01_word_frequencies.csv')
        expected = freq.pivot(index='десятилетие', columns='слово', values='per_million_corpus').loc[PERIODS, words].round(1)
        actual = pd.read_csv(TABLES / 'article1_general_frequencies.csv', index_col=0).loc[PERIODS, words]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_russia_neighbors(self):
        data = pd.read_csv(ROOT / 'results/article2_neighbors/top15_long.csv')
        self.assertEqual(len(data), 540)
        self.assertEqual(data.groupby(['period', 'target']).ngroups, 36)
        self.assertTrue(data.groupby(['period', 'target']).size().eq(15).all())

    def test_russia_differences(self):
        diff = pd.read_csv(TABLES / '05_russia_minus_others.csv')
        for axis in ['liberty', 'civilisation']:
            p = pd.read_csv(TABLES / f'05_projections_{axis}.csv', index_col=0).loc[PERIODS]
            expected = p['russia'] - p.drop(columns='russia').mean(axis=1)
            actual = diff[diff['ось'] == axis].set_index('десятилетие').loc[PERIODS, 'разность']
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
