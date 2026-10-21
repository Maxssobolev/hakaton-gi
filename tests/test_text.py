import unittest
import numpy as np
from ml.text_features import parse_frequencies, transform_records, BINS
from ml.text_experiment import fit_text, text_scores


class TextTests(unittest.TestCase):
    def test_frequency_parsing_and_outcome_exclusion(self):
        values, audit = parse_frequencies('слово 2 Слово 3 банкротство 7 example 0')
        self.assertEqual(values, {'слово': 5})
        self.assertEqual(audit['excluded_occurrences'], {'банкрот': 1})
        self.assertEqual(audit['zero_frequencies'], 1)
        for bad in ('word', 'word -1', 'word 1.5', 'word NaN', 'word 999999999999999999999999999'):
            with self.assertRaises(ValueError): parse_frequencies(bad)

    def test_sparse_transform_independent_of_batch_and_missing(self):
        records = [{'a': 1, 'b': 10}, {}, {'a': 20, 'b': 1}]
        together = transform_records(records)
        self.assertEqual(together.shape, (3, BINS+1))
        self.assertLess(together.nnz, 10)
        self.assertEqual(float(together[1, BINS]), 1.)
        self.assertEqual(float(together[0, BINS]), 0.)
        for i, row in enumerate(records):
            np.testing.assert_array_equal(together[i].toarray(), transform_records([row]).toarray())
        self.assertNotEqual(float(together[0].dot(together[2].T)[0, 0]), 1.)

    def test_training_independent_of_validation_labels(self):
        matrix = transform_records([{'a': 3}, {'b': 4}, {'a': 1}, {}, {'b': 5}, {'a': 8}])
        lookup = np.arange(6); train = np.arange(4); valid = np.array([4, 5])
        labels = np.array([0, 1, 0, 1, 0, 1]); altered = labels.copy(); altered[valid] = 1-altered[valid]
        first = fit_text(matrix, lookup, labels, train, epochs=2)
        second = fit_text(matrix, lookup, altered, train, epochs=2)
        np.testing.assert_array_equal(first.coef_, second.coef_)
        np.testing.assert_array_equal(text_scores(first, matrix, lookup, valid), text_scores(second, matrix, lookup, valid))


if __name__ == '__main__': unittest.main()
