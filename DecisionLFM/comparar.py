"""Compara métricas de execuções com os mesmos IDs, entradas e critérios."""

import argparse
import json
from pathlib import Path


def compare(baseline_dir, decision_dir):
    baseline_dir, decision_dir = Path(baseline_dir), Path(decision_dir)
    runs = [json.loads((p / "run.json").read_text()) for p in [baseline_dir, decision_dir]]
    for field in ["dataset", "dataset_revision", "split", "classes", "questions", "max_length", "warmup", "batch_size"]:
        if runs[0][field] != runs[1][field]:
            raise ValueError(f"Protocolos diferentes no campo {field}")
    if any(r["status"] != "completed" for r in runs):
        raise ValueError("Execução incompleta")
    records = []
    for directory, run in zip([baseline_dir, decision_dir], runs):
        rows = [json.loads(s) for s in (directory / "predictions.jsonl").read_text().splitlines() if s.strip()]
        if len(rows) != run["expected_records"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Quantidade ou IDs inválidos")
        records.append({r["id"]: (r["text"], r["label"]) for r in rows})
    if records[0] != records[1]:
        raise ValueError("As execuções não contêm os mesmos IDs, textos e rótulos")
    metrics = [json.loads((p / "metrics.json").read_text()) for p in [baseline_dir, decision_dir]]
    for run, metric in zip(runs, metrics):
        for field in ["model", "model_revision", "dataset", "dataset_revision", "split"]:
            if metric[field] != run[field]:
                raise ValueError(f"Métricas de outra execução no campo {field}; execute avaliar.py")
        if metric["records"] != run["expected_records"]:
            raise ValueError("Métricas com quantidade de registros diferente da execução")
    fields = ["accuracy", "macro_f1", "brier_multiclass", "latency_median_ms",
              "latency_p95_ms", "peak_allocated_vram_gib", "failed_records"]
    return {
        "records": len(records[0]),
        "regimes": {"CLEF-Flash": runs[0].get("training_regime", "zero_shot"),
                    "DecisionLFM": runs[1]["training_regime"]},
        "metrics": {name: {"CLEF-Flash": metrics[0][name], "DecisionLFM": metrics[1][name]}
                    for name in fields},
        "note": "DecisionLFM foi supervisionado no BANKING77; CLEF-Flash foi avaliado zero-shot. "
                "Versões, codificação e precisões diferem; a latência compara os pipelines executados.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("decision_dir", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline_dir, args.decision_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (args.decision_dir / "comparison.json").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
