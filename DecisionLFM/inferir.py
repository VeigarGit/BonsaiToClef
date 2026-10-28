"""Avalia um DecisionLFM salvo no mesmo protocolo BANKING77 do baseline CLEF."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import dados
from modelo import DecisionLFM, checkpoint_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 0 or args.warmup < 0 or args.max_length < 1:
        parser.error("limit e warmup devem ser >= 0; max-length deve ser > 0")
    if args.output_dir.exists():
        parser.error("output-dir já existe; escolha outro diretório")
    training = json.loads((args.checkpoint / "training.json").read_text())
    if training["status"] != "completed":
        parser.error("Treinamento incompleto: checkpoint recusado para avaliação final")
    torch.set_num_threads(8)
    model = DecisionLFM.load(args.checkpoint)
    model.max_length = args.max_length
    dataset, classes, options = dados.load(args.split)
    if classes != training["classes"] or training["dataset_revision"] != dados.DATASET_REVISION:
        raise ValueError("Dataset diferente do registrado no treino")
    indices = list(range(len(dataset)))
    if args.limit:
        indices = random.Random(args.seed).sample(indices, min(args.limit, len(indices)))

    def predict(text):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = model.decidir(text, dados.QUESTION, options)
        torch.cuda.synchronize()
        result["latency_ms"] = (time.perf_counter() - started) * 1000
        result["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        return result

    print(f"Dataset: {len(indices)} registros; {len(classes)} alternativas.", flush=True)
    for _ in range(args.warmup):
        predict(dataset[indices[0]]["text"])

    args.output_dir.mkdir(parents=True)
    metadata = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "model": "DecisionLFM/LFM3-VL-2B-BANKING77",
        "model_revision": checkpoint_hash(args.checkpoint),
        "checkpoint": str(args.checkpoint.resolve()),
        "base_model": training["base_model"], "base_revision": training["base_revision"],
        "training_regime": "supervised_banking77_train",
        "training_records": len(training["train_indices"]),
        "validation_records": len(training["validation_indices"]),
        "dataset": dados.DATASET, "dataset_revision": dados.DATASET_REVISION,
        "split": args.split, "seed": args.seed, "selected_indices": indices,
        "expected_records": len(indices), "classes": classes,
        "questions": {"intent": {"type": "choice", "instructions": dados.QUESTION, "criteria": options}},
        "max_length": args.max_length, "warmup": args.warmup, "batch_size": 1,
        "dtype": "backbone=bfloat16, head=float32", "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda, "python": sys.version.split()[0],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")},
        "packages": {n: importlib.metadata.version(n) for n in
                     ["torch", "transformers", "datasets", "huggingface_hub", "peft", "accelerate"]},
    }
    metadata_path = args.output_dir / "run.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    completed = failures = 0
    try:
        with (args.output_dir / "predictions.jsonl").open("w") as output:
            for index in indices:
                sample = dataset[index]
                row = {"id": f"banking77:{args.split}:{index}", "text": sample["text"],
                       "label": classes[sample["label"]], "prediction": None,
                       "probabilities": None, "error": None}
                try:
                    row.update(predict(sample["text"]))
                except (RuntimeError, ValueError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    failures += 1
                    torch.cuda.empty_cache()
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                completed += 1
                if completed == 1 or completed % 100 == 0 or completed == len(indices):
                    print(f"{completed}/{len(indices)} registros; {failures} falhas.", flush=True)
        metadata["status"] = "completed"
    finally:
        if metadata["status"] != "completed":
            metadata["status"] = "interrupted"
        metadata["processed_records"] = completed
        metadata["failed_records"] = failures
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
