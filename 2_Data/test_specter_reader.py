"""Small exact-ranking checks; run with python -m unittest discover -s 2_Data."""

import unittest
from collections import Counter
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from specter_reader import (_fingerprint, _top_candidates, deduplicate_works,
                            make_matched_pairs, make_specter_pairs, specter_embeddings)


class SpecterReaderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(32)
        vectors = rng.normal(size=(18, 8)).astype(np.float32)
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.papers = pd.DataFrame({
            'bibcode': [str(i) for i in range(18)],
            'first_author_orcid': [str(i // 3) for i in range(18)],
            'title': [f'title {i}' for i in range(18)],
            'abstract': [f'abstract {i}' for i in range(18)],
        })

    def test_global_ranking_and_shared_limits(self):
        result = make_specter_pairs(self.papers, self.vectors, list('012345'), 10,
                                    paper_limit=2, author_limit=3, block_size=5,
                                    device='cpu')
        # Independent brute force global ranking and greedy selection.
        authors = self.papers.first_author_orcid.tolist()
        candidates = []
        for a in range(18):
            for b in range(a + 1, 18):
                candidates.append((float(self.vectors[a] @ self.vectors[b]), a, b,
                                   int(authors[a] == authors[b])))
        paper_counts, author_counts, label_counts = Counter(), Counter(), Counter()
        expected = []
        for score, a, b, label in sorted(candidates, reverse=True):
            involved = set((authors[a], authors[b]))
            if (label_counts[label] >= 5 or paper_counts[a] >= 2 or paper_counts[b] >= 2
                    or any(author_counts[author] >= 3 for author in involved)):
                continue
            expected.append((str(a), str(b), label))
            paper_counts.update((a, b))
            author_counts.update(involved)
            label_counts[label] += 1
            if len(expected) == 10:
                break
        self.assertEqual(list(result[['bibcode_a', 'bibcode_b', 'label']].itertuples(index=False, name=None)), expected)
        self.assertEqual(label_counts, {0: 5, 1: 5})

    def test_impossible_limits_fail(self):
        with self.assertRaisesRegex(ValueError, 'Cannot fill'):
            make_specter_pairs(self.papers, self.vectors, ['0', '1'], 6,
                               paper_limit=1, author_limit=1, block_size=3, device='cpu')

    def test_cache_fingerprint_changes_with_input(self):
        before = _fingerprint(self.papers, 'model')
        changed = self.papers.copy()
        changed.loc[0, 'abstract'] = 'changed'
        self.assertNotEqual(before, _fingerprint(changed, 'model'))
        self.assertNotEqual(before, _fingerprint(self.papers.iloc[::-1], 'model'))
        self.assertNotEqual(before, _fingerprint(self.papers, 'another-model'))

    def test_cached_embeddings_do_not_load_model(self):
        with TemporaryDirectory() as directory:
            from pathlib import Path
            key = _fingerprint(self.papers, 'model|512|specter-cls-v1')
            np.save(Path(directory) / f'{key}.npy', self.vectors)
            with patch('transformers.AutoModel.from_pretrained', side_effect=AssertionError('loaded model')):
                loaded = specter_embeddings(self.papers, directory, model_name='model', device='cpu')
            np.testing.assert_array_equal(loaded, self.vectors)

    def test_completed_shards_are_reused(self):
        from pathlib import Path
        import json
        with TemporaryDirectory() as directory:
            key = _fingerprint(self.papers, 'model|512|specter-cls-v1')
            shards = Path(directory) / f'{key}-shards-9'
            shards.mkdir()
            np.save(shards / '000000.npy', self.vectors[:9])
            jobs = []

            class FakeProcess:
                def __init__(self, args, **kwargs):
                    import io
                    self.stdout = io.StringIO('')
                    job = json.loads(Path(args[-1]).read_text())
                    jobs.extend(job['shards'])
                    for shard in job['shards']:
                        np.save(shard['output'], self_vectors[9:])

                def wait(self):
                    return 0

                def poll(self):
                    return 0

            self_vectors = self.vectors
            with patch('specter_reader.subprocess.Popen', FakeProcess):
                result = specter_embeddings(self.papers, directory, model_name='model',
                                            devices=['cpu'], shard_size=9)
            self.assertEqual([s['number'] for s in jobs], [1])
            np.testing.assert_array_equal(result, self.vectors)

    def test_two_gpu_ranking_matches_single_gpu(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('two CUDA devices required')
        authors = self.papers.first_author_orcid.to_numpy()
        one, one_counts = _top_candidates(self.vectors, authors, 20, 5, 'cuda:0')
        two, two_counts = _top_candidates(self.vectors, authors, 20, 5, 'cuda:0', ['cuda:0', 'cuda:1'])
        self.assertEqual(one_counts, two_counts)
        for label in (0, 1):
            self.assertEqual([(a, b) for _, a, b in one[label]],
                             [(a, b) for _, a, b in two[label]])

    def test_dedup_removes_conflicts_and_keeps_one_same_author(self):
        titles = ['A sufficiently long shared paper title', 'A sufficiently long shared paper title!',
                  'A different sufficiently long paper title', 'A different sufficiently long paper title',
                  'An independent sufficiently long paper']
        papers = pd.DataFrame({'title': titles, 'abstract': ['different ' + str(i) for i in range(5)],
                               'first_author_orcid': ['a', 'a', 'b', 'c', 'd']})
        keep, stats = deduplicate_works(papers)
        self.assertEqual(keep.tolist(), [0, 4])
        self.assertEqual(stats['duplicate_papers_removed'], 1)
        self.assertEqual(stats['conflicting_author_papers_removed'], 2)

    def test_matched_pairs_are_diverse_balanced_and_reproducible(self):
        import hashlib
        papers = self.papers.copy()
        papers['title'] = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(len(papers))]
        papers['abstract'] = ['text ' + hashlib.sha256(('abstract' + str(i)).encode()).hexdigest()
                              for i in range(len(papers))]
        kwargs = dict(seed=7, paper_limit=2, author_limit=4, device='cpu', batch_size=2)
        result = make_matched_pairs(papers, self.vectors, list('012345'), 6, **kwargs)
        repeat = make_matched_pairs(papers, self.vectors, list('012345'), 6, **kwargs)
        pd.testing.assert_frame_equal(result, repeat)
        self.assertEqual(result.label.value_counts().to_dict(), {0: 3, 1: 3})
        self.assertEqual(result.groupby('match_id').label.sum().tolist(), [1, 1, 1])
        self.assertTrue(all(group.bibcode_a.nunique() == 1 for _, group in result.groupby('match_id')))
        self.assertLessEqual(pd.concat([result.bibcode_a, result.bibcode_b]).value_counts().max(), 2)
        author_by_paper = papers.set_index('bibcode').first_author_orcid.to_dict()
        counts = Counter()
        for row in result.itertuples():
            left, right = author_by_paper[row.bibcode_a], author_by_paper[row.bibcode_b]
            self.assertEqual(int(left == right), row.label)
            counts.update({left, right})
        self.assertLessEqual(max(counts.values()), 4)


if __name__ == '__main__':
    unittest.main()
