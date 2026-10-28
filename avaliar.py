"""Calcula métricas a partir das previsões, sem carregar o modelo ou usar GPU."""

import argparse
import importlib.metadata
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Diretório com run.json e predictions.jsonl")
    args = parser.parse_args()

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    metadata = json.loads((args.run_dir / "run.json").read_text())
    with (args.run_dir / "predictions.jsonl").open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if metadata["status"] != "completed" or len(rows) != metadata["expected_records"]:
        raise SystemExit("Execução incompleta: as métricas finais exigem todos os registros previstos.")
    expected_ids = {f"banking77:{metadata['split']}:{index}" for index in metadata["selected_indices"]}
    if len({row["id"] for row in rows}) != len(rows) or {row["id"] for row in rows} != expected_ids:
        raise SystemExit("IDs ausentes, repetidos ou diferentes dos previstos em run.json.")

    classes = metadata["classes"]
    valid = [row for row in rows if row["error"] is None]
    for row in rows:
        if row["label"] not in classes:
            raise SystemExit(f"Rótulo desconhecido: {row['id']}")
    for row in valid:
        p = row["probabilities"]
        if (
            set(p) != set(classes)
            or not all(np.isfinite(value) and 0 <= value <= 1 for value in p.values())
            or not np.isclose(sum(p.values()), 1, atol=1e-5)
            or row["prediction"] != max(p, key=p.get)
        ):
            raise SystemExit(f"Previsão ou distribuição inválida: {row['id']}")
    labels = [row["label"] for row in rows]
    predictions = [row["prediction"] if row["error"] is None else "__error__" for row in rows]
    brier = [
        sum((row["probabilities"][name] - int(name == row["label"])) ** 2 for name in classes)
        for row in valid
    ]
    latency = [row["latency_ms"] for row in valid]
    confusion = Counter((row["label"], predicted) for row, predicted in zip(rows, predictions)
                        if row["label"] != predicted)
    metrics = {
        "model": metadata["model"],
        "model_revision": metadata["model_revision"],
        "dataset": metadata["dataset"],
        "dataset_revision": metadata["dataset_revision"],
        "split": metadata["split"],
        "evaluation_packages": {
            name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]
        },
        "records": len(rows),
        "failed_records": len(rows) - len(valid),
        "failure_rate": (len(rows) - len(valid)) / len(rows),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
        "brier_multiclass": float(np.mean(brier)) if valid else None,
        "probability_metrics_records": len(valid),
        "latency_median_ms": float(np.median(latency)) if valid else None,
        "latency_p95_ms": float(np.percentile(latency, 95)) if valid else None,
        "peak_allocated_vram_gib": max((row["peak_allocated_vram_gib"] for row in valid), default=None),
        "most_common_errors": [
            {"label": label, "prediction": prediction, "count": count}
            for (label, prediction), count in confusion.most_common(10)
        ],
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.run_dir / "metrics.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
