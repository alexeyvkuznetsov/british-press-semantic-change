import unittest
import numpy as np
from lwm import config as C, tools as T


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = T.load_cache()

    def test_compatible_cache(self):
        self.assertIs(T.validate_cache(self.cache), self.cache)
        self.assertEqual(len(self.cache['words']), 97)
        self.assertEqual(sum(self.cache['corpus_total_words'].values()), 2683235472)

    def test_incompatible_record(self):
        c = dict(self.cache, record=0)
        with self.assertRaises(ValueError):
            T.validate_cache(c)

    def test_missing_period(self):
        c = dict(self.cache, counts=dict(self.cache['counts']))
        del c['counts']['1800s']
        with self.assertRaises(ValueError):
            T.validate_cache(c)

    def test_missing_vector(self):
        c = dict(self.cache, vectors=dict(self.cache['vectors']))
        c['vectors']['1800s'] = dict(c['vectors']['1800s'])
        del c['vectors']['1800s']['empire']
        with self.assertRaises(ValueError):
            T.validate_cache(c)

    def test_field_balance(self):
        V = {'target': np.array([1., 0.]), 'a': np.array([1., 0.]),
             'b': np.array([0., 1.]), 'c': np.array([-1., 0.])}
        self.assertAlmostEqual(T.field_balance(V,'target',['a','b'],['c']), 1.5)

    def test_frequencies(self):
        f = T.frequency_table(self.cache,['empire','imperialism'])
        row = f[(f['слово']=='empire') & (f['десятилетие']=='1800-е')].iloc[0]
        self.assertAlmostEqual(row['per_million_corpus'], 1e6*row['count']/row['corpus_total_words'])
        row = f[(f['слово']=='imperialism') & (f['десятилетие']=='1800-е')].iloc[0]
        self.assertTrue(np.isnan(row['count']))
        self.assertTrue(np.isnan(row['per_million_corpus']))


if __name__ == '__main__':
    unittest.main()
