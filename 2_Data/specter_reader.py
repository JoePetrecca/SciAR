"""SPECTER embeddings and exact, diversity-constrained pair selection."""

import hashlib
import heapq
import json
import re
import subprocess
import sys
import threading
import unicodedata
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


SPECTER_MODEL = 'allenai/specter'


def _normalize_text(value):
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC', str(value)).casefold()).split())


def deduplicate_works(papers):
    """Collapse exact normalized titles/abstracts; remove groups with conflicting authors.

    Grouping occurs before author splitting so detected duplicate works cannot leak
    across splits. Returns positional indices into the original papers and stats.
    """
    n = len(papers)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in ('title', 'abstract'):
        keys = papers[column].map(_normalize_text)
        # Avoid collapsing unrelated short titles or boilerplate abstracts.
        valid = keys.str.len() >= (20 if column == 'title' else 100)
        groups = keys[valid & keys.duplicated(keep=False)].groupby(keys, sort=False).indices
        # groupby indices here are positions relative to the filtered Series.
        positions = np.flatnonzero((valid & keys.duplicated(keep=False)).to_numpy())
        for members in groups.values():
            members = positions[members]
            root = find(int(members[0]))
            for member in members[1:]:
                other = find(int(member))
                if other != root:
                    parent[other] = root
    roots = np.array([find(i) for i in range(n)])
    authors = papers['first_author_orcid'].to_numpy()
    groups = pd.DataFrame({'root': roots, 'author': authors})
    conflict = groups.groupby('root')['author'].transform('nunique').to_numpy() > 1
    first = ~pd.Series(roots).duplicated().to_numpy()
    keep = np.flatnonzero(first & ~conflict)
    stats = {'input_papers': n, 'retained_papers': len(keep),
             'conflicting_author_papers_removed': int(conflict.sum()),
             'duplicate_papers_removed': int((~first & ~conflict).sum())}
    return keep, stats


def _near_duplicate(a, b):
    """Conservative near-duplicate guard for candidate pairs, not corpus-wide detection."""
    for column, threshold in (('title', .90), ('abstract', .95)):
        left, right = a[column], b[column]
        if not left or not right:
            continue
        if min(len(left), len(right)) / max(len(left), len(right)) < threshold:
            continue
        comparison = SequenceMatcher(None, left, right, autojunk=False)
        if comparison.quick_ratio() >= threshold and comparison.ratio() >= threshold:
            return True
    return False


def make_matched_pairs(papers, embeddings, author_ids, n_pairs, *, seed=42,
                       paper_limit=2, author_limit=10, device=None, batch_size=16):
    """Sample varied distinct positives and similarity-match one negative per positive.

    Each negative reuses its positive's anchor. The closest eligible different-author
    paper is chosen by absolute distance to the positive's SPECTER cosine similarity.
    Pair/author limits are shared and positive anchors are reserved in advance.
    """
    if n_pairs < 2 or n_pairs % 2 or paper_limit < 2 or author_limit < 2 or batch_size < 1:
        raise ValueError('Need an even pair count, paper limit >= 2 and author limit >= 2')
    indices = np.flatnonzero(papers['first_author_orcid'].isin(author_ids).to_numpy())
    subset = papers.iloc[indices].reset_index(drop=True)
    vectors = np.asarray(embeddings[indices], dtype=np.float32)
    authors = subset['first_author_orcid'].to_numpy()
    codes, _ = pd.factorize(authors, sort=True)
    normalized = [{'title': _normalize_text(t), 'abstract': _normalize_text(a)}
                  for t, a in zip(subset.title, subset.abstract)]
    groups = pd.Series(np.arange(len(subset))).groupby(authors, sort=True).agg(list)
    groups = {key: np.asarray(value) for key, value in groups.items() if len(value) >= 2}
    rng = np.random.default_rng(seed)
    quota = n_pairs // 2
    selected, paper_counts, author_counts = [], np.zeros(len(subset), dtype=np.int32), {}
    eligible = list(groups)
    while len(selected) < quota and eligible:
        rng.shuffle(eligible)
        next_eligible = []
        for author in eligible:
            if len(selected) == quota:
                break
            if author_counts.get(author, 0) + 2 > author_limit:
                continue
            available = groups[author][paper_counts[groups[author]] == 0].copy()
            rng.shuffle(available)
            chosen = None
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    a, b = int(available[i]), int(available[j])
                    if not _near_duplicate(normalized[a], normalized[b]):
                        chosen = (a, b)
                        break
                if chosen:
                    break
            if chosen is None:
                continue
            a, b = chosen
            paper_counts[a] += 1
            paper_counts[b] += 1
            # Reserve one additional appearance for this author's negative anchor.
            author_counts[author] = author_counts.get(author, 0) + 2
            selected.append((a, b, float(vectors[a] @ vectors[b])))
            if len(available) >= 4 and author_counts[author] + 2 <= author_limit:
                next_eligible.append(author)
        eligible = next_eligible
    if len(selected) != quota:
        raise ValueError(f'Only found {len(selected)} distinct positive pairs of {quota}; '
                         'reduce pair count or relax author limit.')

    reserved_anchors = {a for a, _, _ in selected}
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    x = torch.as_tensor(vectors.copy(), dtype=torch.float32, device=device)
    code_tensor = torch.as_tensor(codes, device=device)
    excluded = torch.zeros(len(subset), dtype=torch.bool, device=device)
    if reserved_anchors:
        excluded[list(reserved_anchors)] = True
    # Authors with no remaining capacity cannot be selected as negative partners.
    author_used = {key: value for key, value in author_counts.items()}
    negative_pairs = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in tqdm(range(0, quota, batch_size), desc='Matching negative topics'):
            batch = selected[start:start + batch_size]
            anchors = torch.tensor([a for a, _, _ in batch], device=device)
            similarities = x[anchors] @ x.T
            for row, (a, b, target) in enumerate(batch):
                distances = torch.abs(similarities[row] - target)
                distances[code_tensor == int(codes[a])] = float('inf')
                distances[excluded] = float('inf')
                while True:
                    candidate = int(torch.argmin(distances).item())
                    distance = float(distances[candidate].item())
                    if not np.isfinite(distance):
                        raise ValueError(f'Could not find eligible negative for positive {start + row}; '
                                         'reduce pair count or relax limits.')
                    other_author = authors[candidate]
                    if (paper_counts[candidate] >= paper_limit
                            or author_used.get(other_author, 0) >= author_limit
                            or _near_duplicate(normalized[a], normalized[candidate])):
                        distances[candidate] = float('inf')
                        continue
                    paper_counts[a] += 1
                    paper_counts[candidate] += 1
                    author_used[other_author] = author_used.get(other_author, 0) + 1
                    if paper_counts[candidate] >= paper_limit:
                        excluded[candidate] = True
                    negative_pairs.append((a, candidate, float(similarities[row, candidate].item())))
                    break
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    rows = []
    for pair_id, ((a, b, positive_score), (_, c, negative_score)) in enumerate(zip(selected, negative_pairs)):
        for left_index, right_index, label, score in ((a, b, 1, positive_score), (a, c, 0, negative_score)):
            left, right = subset.iloc[left_index], subset.iloc[right_index]
            rows.append({'text_a': left.title + '\n\n' + left.abstract,
                         'text_b': right.title + '\n\n' + right.abstract,
                         'label': label, 'bibcode_a': left.bibcode, 'bibcode_b': right.bibcode,
                         'specter_similarity': score, 'match_id': pair_id})
    result = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f'Selected {len(result):,} pairs, {int((paper_counts > 0).sum()):,} papers, '
          f'{len(author_used):,} authors')
    print(result.groupby('label')['specter_similarity'].quantile([0, .25, .5, .75, 1]).unstack().to_string())
    gaps = np.abs(np.array([s for _, _, s in selected]) - np.array([s for _, _, s in negative_pairs]))
    print(f'Absolute positive/negative similarity gap: median={np.median(gaps):.4f}, '
          f'95th percentile={np.quantile(gaps, .95):.4f}, max={gaps.max():.4f}')
    return result


def _fingerprint(papers, model_name):
    digest = hashlib.sha256(model_name.encode())
    for column in ('bibcode', 'first_author_orcid', 'title', 'abstract'):
        digest.update(pd.util.hash_pandas_object(papers[column], index=False).values.tobytes())
    return digest.hexdigest()


def _encode_shards(job_path):
    """Subprocess entry point: load one model per device and save each completed shard."""
    from transformers import AutoModel, AutoTokenizer
    job = json.loads(Path(job_path).read_text())
    device = torch.device(job['device'])
    tokenizer = AutoTokenizer.from_pretrained(job['model'])
    model = AutoModel.from_pretrained(job['model']).to(device).eval()
    if job['precision'] == 'float16':
        model = model.half()
    for shard in job['shards']:
        output = Path(shard['output'])
        if output.exists():
            continue
        batch_data = pd.read_parquet(shard['input'])
        print(f"Encoding SPECTER shard {shard['number']} on {device} ({len(batch_data)} papers)", flush=True)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(batch_data), job['batch_size']):
                batch = batch_data.iloc[start:start + job['batch_size']]
                texts = (batch['title'] + tokenizer.sep_token + batch['abstract']).tolist()
                tokens = tokenizer(texts, padding=True, truncation=True,
                                   max_length=job['max_length'], return_tensors='pt').to(device)
                vectors = model(**tokens).last_hidden_state[:, 0, :].float()
                batches.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy())
        tmp = output.with_suffix('.tmp.npy')
        np.save(tmp, np.concatenate(batches).astype(np.float32, copy=False))
        tmp.replace(output)
        print(f"Saved SPECTER shard {shard['number']} on {device}", flush=True)


def specter_embeddings(papers, cache_dir, model_name=SPECTER_MODEL, batch_size=32,
                       max_length=512, device=None, devices=None, precision='float32',
                       shard_size=8192):
    """Encode in resumable shards; cache is keyed by ordered input and precision."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if precision not in ('float32', 'float16') or batch_size < 1 or shard_size < 1:
        raise ValueError('Invalid SPECTER precision, batch size, or shard size')
    devices = [str(d) for d in (devices or [device or ('cuda' if torch.cuda.is_available() else 'cpu')])]
    if len(set(devices)) != len(devices):
        raise ValueError('SPECTER devices must be distinct')
    if precision == 'float16' and any(torch.device(d).type != 'cuda' for d in devices):
        raise ValueError('float16 SPECTER encoding requires CUDA; use float32 on CPU')
    # Retain compatibility with existing full precision caches.
    version = 'specter-cls-v1' if precision == 'float32' else f'specter-cls-v2-{precision}'
    fingerprint = _fingerprint(papers, f'{model_name}|{max_length}|{version}')
    cache = cache_dir / f'{fingerprint}.npy'
    if cache.exists():
        embeddings = np.load(cache, mmap_mode='r')
        if embeddings.ndim != 2 or len(embeddings) != len(papers):
            raise ValueError(f'Invalid SPECTER cache: {cache}')
        print(f'Using SPECTER cache: {cache}')
        return embeddings

    shard_dir = cache_dir / f'{fingerprint}-shards-{shard_size}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for number, start in enumerate(range(0, len(papers), shard_size)):
        input_path = shard_dir / f'{number:06d}.parquet'
        output_path = shard_dir / f'{number:06d}.npy'
        expected = min(shard_size, len(papers) - start)
        if output_path.exists():
            cached = np.load(output_path, mmap_mode='r')
            if cached.ndim != 2 or len(cached) != expected:
                raise ValueError(f'Invalid SPECTER shard: {output_path}')
        else:
            papers.iloc[start:start + shard_size][['title', 'abstract']].to_parquet(input_path, index=False)
        shards.append({'number': number, 'input': str(input_path.resolve()),
                       'output': str(output_path.resolve())})
    assignments = [[] for _ in devices]
    for index, shard in enumerate(s for s in shards if not Path(s['output']).exists()):
        assignments[index % len(devices)].append(shard)
    processes, log_threads = [], []
    def forward_output(stream):
        for line in stream:
            print(line, end='', flush=True)
        stream.close()
    try:
        for index, (worker_device, assigned) in enumerate(zip(devices, assignments)):
            if not assigned:
                continue
            job = {'device': worker_device, 'model': model_name, 'precision': precision,
                   'batch_size': batch_size, 'max_length': max_length, 'shards': assigned}
            job_path = shard_dir / f'worker-{index}.json'
            job_path.write_text(json.dumps(job))
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                        '--embed-worker', str(job_path.resolve())],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            processes.append(process)
            thread = threading.Thread(target=forward_output, args=(process.stdout,), daemon=True)
            thread.start()
            log_threads.append(thread)
        failures = [p.wait() for p in processes]
        for thread in log_threads:
            thread.join()
        if any(code != 0 for code in failures):
            raise RuntimeError(f'SPECTER worker failed: exit codes {failures}; completed shards can be reused')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for thread in log_threads:
            thread.join()
    vectors = np.concatenate([np.load(s['output']) for s in shards]).astype(np.float32, copy=False)
    temporary = cache.with_suffix('.tmp.npy')
    np.save(temporary, vectors)
    temporary.replace(cache)
    cache.with_suffix('.json').write_text(json.dumps({
        'model': model_name, 'max_length': max_length, 'papers': len(papers),
        'fingerprint': fingerprint, 'pooling': 'CLS', 'normalized': True, 'precision': precision,
    }, indent=2))
    for shard in shards:
        Path(shard['input']).unlink(missing_ok=True)
        Path(shard['output']).unlink(missing_ok=True)
    return vectors


def _push_candidates(heap, candidates, limit):
    for candidate in candidates:
        if len(heap) < limit:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)


def _positive_candidates(vectors, authors, limit, block_size):
    """Only compare papers belonging to the same author."""
    groups = pd.Series(np.arange(len(authors))).groupby(authors, sort=False).agg(list)
    heap, count = [], 0
    for group in groups:
        ids = np.asarray(group)
        for start in range(0, len(ids), block_size):
            left = ids[start:start + block_size]
            for offset in range(start, len(ids), block_size):
                right = ids[offset:offset + block_size]
                scores = vectors[left] @ vectors[right].T
                mask = np.ones(scores.shape, dtype=bool)
                if start == offset:
                    mask = np.triu(mask, 1)
                rows, cols = np.nonzero(mask)
                count += len(rows)
                values = scores[rows, cols]
                if len(heap) == limit:
                    keep = values >= heap[0][0]
                    rows, cols, values = rows[keep], cols[keep], values[keep]
                if len(values) > limit:
                    chosen = np.argpartition(values, -limit)[-limit:]
                    rows, cols, values = rows[chosen], cols[chosen], values[chosen]
                _push_candidates(heap, ((float(s), int(left[r]), int(right[c]))
                                        for s, r, c in zip(values, rows, cols)), limit)
    return heap, count


def _negative_candidates(vectors, author_codes, limit, block_size, device, starts):
    x = torch.as_tensor(np.asarray(vectors).copy(), dtype=torch.float32, device=device)
    codes = torch.as_tensor(author_codes, device=device)
    heap, count, n = [], 0, len(x)
    for i in tqdm(starts, desc=f'Negative blocks {device}'):
        i_end = min(i + block_size, n)
        for j in range(i, n, block_size):
            j_end = min(j + block_size, n)
            scores = x[i:i_end] @ x[j:j_end].T
            valid = codes[i:i_end, None] != codes[None, j:j_end]
            if i == j:
                valid = torch.triu(valid, diagonal=1)
            count += int(valid.sum().item())
            if len(heap) == limit:
                valid &= scores >= heap[0][0]
            positions = valid.flatten().nonzero().flatten()
            k = min(limit, positions.numel())
            if not k:
                continue
            values, chosen = torch.topk(scores.flatten()[positions], k, sorted=False)
            positions = positions[chosen]
            rows = (positions // (j_end - j) + i).cpu().tolist()
            cols = (positions % (j_end - j) + j).cpu().tolist()
            _push_candidates(heap, zip(values.cpu().tolist(), rows, cols), limit)
    return heap, count


def _top_candidates(vectors, authors, limit, block_size, device, devices=None):
    """Return top unordered pairs per label; distribute negative blocks over devices."""
    devices = [torch.device(d) for d in (devices or [device])]
    codes = pd.factorize(authors, sort=True)[0]
    starts = list(range(0, len(vectors), block_size))
    # Interleaving gives each worker a mixture of long and short block rows.
    assignments = [starts[i::len(devices)] for i in range(len(devices))]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with ThreadPoolExecutor(max_workers=len(devices) + 1) as executor:
            positive = executor.submit(_positive_candidates, vectors, authors, limit, block_size)
            negatives = [executor.submit(_negative_candidates, vectors, codes, limit, block_size, d, rows)
                         for d, rows in zip(devices, assignments) if rows]
            pos_heap, pos_count = positive.result()
            neg_heap, neg_count = [], 0
            for future in negatives:
                heap, count = future.result()
                _push_candidates(neg_heap, heap, limit)
                neg_count += count
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return {0: sorted(neg_heap, reverse=True), 1: sorted(pos_heap, reverse=True)}, {0: neg_count, 1: pos_count}


def make_specter_pairs(papers, embeddings, author_ids, n_pairs, *, paper_limit=2,
                       author_limit=10, block_size=2048, device=None, devices=None):
    """Rank all within-split pairs and greedily enforce shared diversity limits.

    Candidates are merged by score across both labels; each label has a fixed quota.
    If a truncated category is exhausted before its quota is filled, rerun with a
    larger pool. This preserves the result of scanning the complete global ranking.
    """
    if n_pairs < 2 or n_pairs % 2 or min(paper_limit, author_limit, block_size) < 1:
        raise ValueError('Pair count must be positive and even; limits must be positive')
    indices = np.flatnonzero(papers['first_author_orcid'].isin(author_ids).to_numpy())
    subset = papers.iloc[indices].reset_index(drop=True)
    vectors = np.asarray(embeddings[indices])
    authors = subset['first_author_orcid'].to_numpy()
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    quota = n_pairs // 2
    limit = max(n_pairs * 4, 1000)
    while True:
        candidates, counts = _top_candidates(vectors, authors, limit, block_size, device, devices)
        merged = sorted(((score, a, b, label) for label, items in candidates.items()
                         for score, a, b in items), reverse=True)
        used_papers, used_authors = {}, {}
        selected, selected_counts = [], {0: 0, 1: 0}
        # If a truncated list ends, do not process lower scores from the other
        # category: unseen candidates may have higher scores and affect limits.
        cutoff = max((items[-1][0] for label, items in candidates.items()
                      if len(items) < counts[label] and items), default=-float('inf'))
        for score, a, b, label in merged:
            if score < cutoff:
                break
            if selected_counts[label] == quota:
                continue
            aa, ab = authors[a], authors[b]
            author_increments = {aa: 1} if aa == ab else {aa: 1, ab: 1}
            if (used_papers.get(a, 0) >= paper_limit or used_papers.get(b, 0) >= paper_limit
                    or any(used_authors.get(key, 0) + value > author_limit
                           for key, value in author_increments.items())):
                continue
            used_papers[a] = used_papers.get(a, 0) + 1
            used_papers[b] = used_papers.get(b, 0) + 1
            for key, value in author_increments.items():
                used_authors[key] = used_authors.get(key, 0) + value
            selected.append((score, a, b, label))
            selected_counts[label] += 1
            if all(count == quota for count in selected_counts.values()):
                break
        if all(count == quota for count in selected_counts.values()):
            break
        if all(len(candidates[label]) == counts[label] for label in (0, 1)):
            raise ValueError(f'Cannot fill balanced pair quotas with greedy selection and diversity limits: '
                             f'{selected_counts}; available pairs: {counts}. '
                             'Increase limits or reduce pair count.')
        limit *= 2
        print(f'Expanding exact candidate pool to {limit:,} per label')

    rows = []
    for score, a, b, label in selected:
        left, right = subset.iloc[a], subset.iloc[b]
        rows.append({'text_a': left['title'] + '\n\n' + left['abstract'],
                     'text_b': right['title'] + '\n\n' + right['abstract'],
                     'label': label, 'bibcode_a': left['bibcode'],
                     'bibcode_b': right['bibcode'], 'specter_similarity': score})
    result = pd.DataFrame(rows)
    print(f'Selected {len(result):,} pairs; {len(used_papers):,} papers; '
          f'{len(used_authors):,} authors; SPECTER similarity '
          f'{result.specter_similarity.min():.4f}–{result.specter_similarity.max():.4f}')
    print(result.groupby('label')['specter_similarity'].quantile([0, .25, .5, .75, 1]).unstack().to_string())
    return result


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--embed-worker':
        raise SystemExit('This module is called by the training notebook.')
    _encode_shards(sys.argv[2])
