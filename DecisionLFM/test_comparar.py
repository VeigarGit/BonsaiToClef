"""A comparação recusa execuções com amostras ou protocolos diferentes."""

import json
import tempfile
import unittest
from pathlib import Path

from comparar import compare


class CompareTests(unittest.TestCase):
    def test_matching_records_and_rejected_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / name for name in ["baseline", "decision"]]
            run = {"dataset": "test", "dataset_revision": "sha", "split": "test",
                   "classes": ["a", "b"], "questions": {}, "max_length": 100, "warmup": 3,
                   "status": "completed", "expected_records": 1, "training_regime": "supervised", "batch_size": 1,
                   "model": "test-model", "model_revision": "test-revision"}
            metrics = {k: 0 for k in ["accuracy", "macro_f1", "brier_multiclass", "latency_median_ms",
                                      "latency_p95_ms", "peak_allocated_vram_gib", "failed_records"]}
            metrics.update({k: run[k] for k in ["model", "model_revision", "dataset", "dataset_revision", "split"]})
            metrics["records"] = 1
            row = {"id": "one", "text": "same input", "label": "a"}
            for d in dirs:
                d.mkdir()
                (d / "run.json").write_text(json.dumps(run))
                (d / "metrics.json").write_text(json.dumps(metrics))
                (d / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(compare(*dirs)["records"], 1)
            metrics["model_revision"] = "stale-revision"
            (dirs[1] / "metrics.json").write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "Métricas de outra execução"):
                compare(*dirs)
            metrics["model_revision"] = run["model_revision"]
            (dirs[1] / "metrics.json").write_text(json.dumps(metrics))
            row["text"] = "different input"
            (dirs[1] / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "mesmos IDs"):
                compare(*dirs)
            (dirs[1] / "predictions.jsonl").write_text(json.dumps({**row, "text": "same input"}) + "\n")
            run["dataset_revision"] = "another revision"
            (dirs[1] / "run.json").write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, "dataset_revision"):
                compare(*dirs)


if __name__ == "__main__":
    unittest.main()
