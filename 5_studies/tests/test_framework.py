"""Offline tests: tiny real Transformers architectures and full local training pipeline."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertModel, BertTokenizerFast,
                          ModernBertConfig, ModernBertModel, RobertaConfig, RobertaModel)

import train
from collect_results import author_components, collect, paired_bootstrap
from common import (SPLITS, configuration, experiments, load_dataset, read_json,
                    save_manifest, select_experiment, validate_pairs, write_json)
from engine import Selection, attach_lora, encode
from engine import device_for
from verify import check_report
from prepare_data import existing_readers, subset_pairs
import prepare_data


def fixture(directory):
    papers, pairs, assignments = [], {}, {}
    for split in SPLITS:
        authors = [f"{split}-author-{i}" for i in range(4)]
        assignments[split] = authors
        for i, author in enumerate(authors):
            for j in range(2):
                papers.append(dict(bibcode=f"{split}-{i}-{j}", first_author_orcid=author,
                                   title=f"paper {i} {j}", abstract=f"science text {split} {i} {j}"))
        rows = []
        for i in range(2):
            for label in (1, 0):
                other = i if label else i+2
                rows.append(dict(pair_id=f"{split}:{len(rows)}", bibcode_a=f"{split}-{i}-0",
                                 bibcode_b=f"{split}-{other}-1", author_a=authors[i],
                                 author_b=authors[other], label=label, match_id=i,
                                 specter_similarity=.7))
        pairs[split] = pd.DataFrame(rows)
    papers = pd.DataFrame(papers)
    directory.mkdir()
    papers.to_parquet(directory / "papers.parquet", index=False)
    write_json(directory / "author_splits.json", assignments)
    for split, frame in pairs.items():
        frame.to_parquet(directory / f"{split}.parquet", index=False)
    save_manifest(directory, dict(dataset_seed=67, sizes={s: len(f) for s, f in pairs.items()}))
    return pairs, papers, assignments


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_array_tasks(self):
        self.assertEqual(len(experiments()), 15)
        self.assertEqual(len(set(experiments())), 15)
        for i, run in enumerate(experiments()):
            self.assertEqual(select_experiment(i), run)
        for i in (-1, 15):
            with self.assertRaises(ValueError):
                select_experiment(i)

    def test_dataset_integrity_and_shared_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            pairs, papers, assignments = fixture(path)
            fingerprints = [load_dataset(path)[2]["fingerprint"] for _ in experiments()]
            self.assertEqual(len(set(fingerprints)), 1)
            bad = copy.deepcopy(pairs)
            bad["test"].loc[0, "author_a"] = assignments["train"][0]
            with self.assertRaises(ValueError):
                validate_pairs(bad, papers, assignments)
            with (path / "train.parquet").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_dataset(path)

    def test_selection_and_early_stopping(self):
        selection = Selection(3)
        self.assertTrue(selection.update(.7, 1))
        self.assertFalse(selection.update(.7, 2))
        self.assertTrue(selection.update(.8, 3))
        for epoch in (4, 5, 6):
            selection.update(.79, epoch)
        self.assertTrue(selection.stopped)
        self.assertEqual(selection.best_epoch, 3)

    def test_cuda_required_and_cpu_report_rejected(self):
        with patch("engine.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                device_for()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cpu.json"
            write_json(path, dict(passed=True, device="cpu"))
            with self.assertRaisesRegex(RuntimeError, "GPU verification"):
                check_report(path)

    def test_three_real_architectures(self):
        specs, config = configuration()
        bases = [
            ("bert", BertModel(BertConfig(vocab_size=30, hidden_size=16, num_hidden_layers=2,
                                         num_attention_heads=2, intermediate_size=32))),
            ("roberta", RobertaModel(RobertaConfig(vocab_size=30, hidden_size=16, num_hidden_layers=2,
                                                  num_attention_heads=2, intermediate_size=32,
                                                  max_position_embeddings=64))),
            ("modernbert", ModernBertModel(ModernBertConfig(
                vocab_size=30, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                intermediate_size=32, max_position_embeddings=64, local_attention=16,
                reference_compile=False, pad_token_id=0, eos_token_id=2,
                bos_token_id=1, cls_token_id=1, sep_token_id=2,
                _attn_implementation="eager"))),
        ]
        for name, base in bases:
            with self.subTest(name=name):
                model, info = attach_lora(base, specs[name], config)
                model.eval()
                embeddings = encode(model, dict(input_ids=torch.tensor([[1, 5, 2, 0], [1, 6, 7, 2]]),
                                                attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])))
                self.assertEqual(embeddings.shape, (2, 16))
                torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
                loss = torch.nn.functional.cosine_embedding_loss(embeddings[:1], embeddings[1:],
                                                                 torch.ones(1), margin=.2)
                loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in model.named_parameters() if "lora_" in n))
                self.assertGreater(info["trainable_parameters"], 0)

    def test_existing_sampler_and_nested_subsets(self):
        reader, specter = existing_readers()
        rng = np.random.default_rng(67)
        papers = pd.DataFrame(dict(
            bibcode=[str(i) for i in range(30)], first_author_orcid=[str(i//3) for i in range(30)],
            title=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(30)],
            abstract=[hashlib.sha256(f"text-{i}".encode()).hexdigest() for i in range(30)]))
        vectors = rng.normal(size=(30, 8)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        a = specter.make_matched_pairs(papers, vectors, papers.first_author_orcid.unique(),
                                      12, seed=67, device="cpu")
        b = specter.make_matched_pairs(papers, vectors, papers.first_author_orcid.unique(),
                                      12, seed=67, device="cpu")
        pd.testing.assert_frame_equal(a, b)
        small, medium = subset_pairs(a, 4), subset_pairs(a, 8)
        self.assertTrue(set(small.match_id) <= set(medium.match_id))
        self.assertEqual(small.label.sum(), 2)

    def test_prepare_cli_with_cached_specter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, specter = existing_readers()
            papers = pd.DataFrame(dict(
                bibcode=[str(i) for i in range(180)],
                first_author_orcid=[str(i//3) for i in range(180)],
                title=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(180)],
                abstract=[hashlib.sha256(f"abstract-{i}".encode()).hexdigest() for i in range(180)]))
            source = root / "papers.parquet"
            papers.to_parquet(source, index=False)
            cache = root / "cache"
            cache.mkdir()
            vectors = np.random.default_rng(67).normal(size=(180, 8)).astype(np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            key = specter._fingerprint(papers, "allenai/specter|512|specter-cls-v1")
            np.save(cache / f"{key}.npy", vectors)
            out = root / "fixed"
            argv = ["prepare_data.py", "--papers", str(source), "--output", str(out),
                    "--cache", str(cache), "--device", "cpu", "--train-pairs", "12",
                    "--validation-pairs", "4", "--test-pairs", "4", "--precision", "float32"]
            with patch.object(sys, "argv", argv):
                prepare_data.main()
            pairs, _, manifest = load_dataset(out)
            self.assertEqual(manifest["sizes"], dict(train=12, validation=4, test=4))
            subset = root / "subset"
            with patch.object(sys, "argv", ["prepare_data.py", "--subset-of", str(out),
                                          "--output", str(subset), "--train-pairs", "4",
                                          "--validation-pairs", "4", "--test-pairs", "4"]):
                prepare_data.main()
            child, _, _ = load_dataset(subset)
            pd.testing.assert_frame_equal(pairs["test"], child["test"])
            self.assertEqual((out / "test.parquet").read_bytes(), (subset / "test.parquet").read_bytes())
            # Extension uses the retained corpus, which is unchanged for this synthetic fixture.
            extended = root / "extended"
            with patch.object(sys, "argv", ["prepare_data.py", "--extend-from", str(out),
                                          "--output", str(extended), "--cache", str(cache),
                                          "--device", "cpu", "--train-pairs", "16",
                                          "--validation-pairs", "4", "--test-pairs", "4",
                                          "--precision", "float32"]):
                prepare_data.main()
            extension, _, manifest = load_dataset(extended)
            self.assertEqual(len(extension["train"]), 16)
            self.assertEqual(manifest["operation"], "extend_train_fixed_evaluation")
            self.assertEqual((out / "test.parquet").read_bytes(), (extended / "test.parquet").read_bytes())

    def test_component_bootstrap(self):
        frame = pd.DataFrame(dict(author_a=["a", "a", "c", "c", "e", "e"],
                                  author_b=["a", "b", "c", "d", "e", "f"],
                                  label=[1, 0, 1, 0, 1, 0]))
        self.assertEqual(len(set(author_components(frame))), 3)
        scores = np.array([.8, .2, .7, .1, .9, .3])
        result = paired_bootstrap(frame, [scores]*3, [scores]*3, 20)
        self.assertEqual(result["interval_95"], [0., 0.])
        frame["author_b"] = "shared"
        self.assertIsNone(paired_bootstrap(frame, [scores], [scores], 20)["interval_95"])

    def test_training_resume_exports_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            fixture(data)
            checkpoint = root / "tiny"
            checkpoint.mkdir()
            (checkpoint / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npaper\nscience\ntext\n0\n1\n2\n3\ntrain\nvalidation\ntest\n")
            tokenizer = BertTokenizerFast(vocab_file=str(checkpoint / "vocab.txt"))
            tokenizer.save_pretrained(checkpoint)
            BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=32)).save_pretrained(checkpoint)
            _, defaults = configuration()
            config = defaults | dict(epochs=2, workers=0, microbatch_size=1, batch_size=2, max_length=16)
            models = {"bert": dict(checkpoint=str(checkpoint), targets=["query", "value"], modules_per_layer=2)}
            out = root / "outputs"
            argv = ["train.py", "--model", "bert", "--seed", "67", "--data", str(data),
                    "--output-root", str(out), "--cpu-smoke"]
            real_epoch = train.train_epoch
            count = 0
            def interrupt(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("simulated interruption")
                return real_epoch(*args, **kwargs)
            with patch.object(train, "configuration", return_value=(models, config)):
                with patch.object(sys, "argv", argv), patch.object(train, "train_epoch", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        train.main()
                with patch.object(sys, "argv", argv + ["--resume"]):
                    train.main()
                # Compare with uninterrupted training, including shuffle/dropout RNG.
                out2 = root / "uninterrupted"
                argv2 = argv.copy()
                argv2[argv2.index(str(out))] = str(out2)
                with patch.object(sys, "argv", argv2):
                    train.main()
            run = out / "bert/seed_67"
            full = out2 / "bert/seed_67"
            pd.testing.assert_frame_equal(pd.read_csv(run / "finetuned_test_scores.csv"),
                                          pd.read_csv(full / "finetuned_test_scores.csv"))
            columns = ["epoch", "train_loss", "val_loss", "val_auc"]
            pd.testing.assert_frame_equal(pd.read_csv(run / "history.csv")[columns],
                                          pd.read_csv(full / "history.csv")[columns])
            self.assertEqual(len(pd.read_csv(run / "history.csv")), 2)
            for name in ("loss.png", "validation_auc.png", "test_roc.png", "test_distributions.png",
                         "confusion_matrices.png", "adapter/adapter_model.safetensors"):
                self.assertTrue((run / name).exists(), name)
            import shutil
            for model, seed in experiments():
                target = out / model / f"seed_{seed}"
                if target == run:
                    continue
                shutil.copytree(run, target)
                metadata = read_json(target / "metrics.json")
                metadata.update(model=model, seed=seed)
                write_json(target / "metrics.json", metadata)
            individual, summary = collect(out, root / "comparison", 10, allow_incomplete=True)
            self.assertEqual(len(individual), 15)
            self.assertTrue((summary.runs == 3).all())
            self.assertTrue((summary.std_test_auc == 0).all())
            self.assertEqual(len(read_json(root / "comparison/bootstrap.json")["comparisons"]), 10)


if __name__ == "__main__":
    unittest.main()
