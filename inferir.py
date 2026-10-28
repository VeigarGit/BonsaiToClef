"""Executa CLEF-Flash no BANKING77, sem treinamento e sem exemplos no prompt."""

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL = "Cloudflare/clef-flash"
MODEL_REVISION = "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
DATASET = "PolyAI/banking77"
DATASET_REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 executa todo o split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0 or args.warmup < 0 or args.max_length < 1:
        parser.error("limit e warmup devem ser >= 0; max-length deve ser > 0")
    if args.output_dir.exists():
        parser.error("output-dir já existe; escolha outro diretório para preservar os resultados")

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("Este experimento requer uma GPU CUDA com suporte a BF16.")

    # Os dois repositórios são públicos; não depender de credenciais locais.
    api = HfApi(token=False)
    model_revision = api.model_info(MODEL, revision=args.model_revision).sha
    dataset_revision = api.dataset_info(DATASET, revision=args.dataset_revision).sha
    dataset = load_dataset(DATASET, split=args.split, revision=dataset_revision, token=False)
    classes = dataset.features["label"].names
    indices = list(range(len(dataset)))
    if args.limit:
        indices = random.Random(args.seed).sample(indices, min(args.limit, len(indices)))
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "Identify the customer's banking intent. Choose exactly one option.",
            "criteria": {name: name.replace("_", " ").replace("?", "") for name in classes},
        }
    }
    print(f"Dataset: {len(indices)} registros; {len(classes)} alternativas por registro.", flush=True)
    model_path = snapshot_download(MODEL, revision=model_revision, token=False)
    sys.path.insert(0, model_path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    model, processor = load_release_model(model_path, device="cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def predict(text):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        record = {"state": text, "questions": questions}
        # Codificar a entrada inteira permite recusar excesso de tokens sem truncar o texto.
        encoded = encode_record(processor.tokenizer, record, processor=processor, max_length=sys.maxsize)
        if len(encoded.input_ids) > args.max_length:
            raise ValueError(f"Entrada com {len(encoded.input_ids)} tokens; limite {args.max_length}.")
        batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
        logits = model(batch)[0][0]
        probabilities = dict(zip(encoded.questions[0].option_ids, logits.float().softmax(-1).tolist()))
        prediction = max(probabilities, key=probabilities.get)
        torch.cuda.synchronize()
        return {
            "prediction": prediction,
            "confidence": probabilities[prediction],
            "probabilities": probabilities,
            "input_tokens": len(encoded.input_ids),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }

    print(f"Aquecimento: {args.warmup} chamadas (fora das métricas).", flush=True)
    for _ in range(args.warmup):
        predict(dataset[indices[0]]["text"])

    args.output_dir.mkdir(parents=True)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "model": MODEL,
        "model_revision": model_revision,
        "dataset": DATASET,
        "dataset_revision": dataset_revision,
        "split": args.split,
        "seed": args.seed,
        "selected_indices": indices,
        "expected_records": len(indices),
        "classes": classes,
        "questions": questions,
        "max_length": args.max_length,
        "warmup": args.warmup,
        "batch_size": 1,
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "datasets", "huggingface_hub", "accelerate"]
        },
    }
    metadata_path = args.output_dir / "run.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    completed = failures = 0
    try:
        with (args.output_dir / "predictions.jsonl").open("w") as output:
            for index in indices:
                sample = dataset[index]
                row = {
                    "id": f"banking77:{args.split}:{index}",
                    "text": sample["text"],
                    "label": classes[sample["label"]],
                    "prediction": None,
                    "probabilities": None,
                    "error": None,
                }
                try:
                    row.update(predict(sample["text"]))
                except (RuntimeError, ValueError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    failures += 1
                    torch.cuda.empty_cache()
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                completed += 1
                if completed == 1 or completed % 20 == 0 or completed == len(indices):
                    print(f"{completed}/{len(indices)} registros; {failures} falhas.", flush=True)
        metadata["status"] = "completed"
    finally:
        metadata["processed_records"] = completed
        metadata["failed_records"] = failures
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        if metadata["status"] != "completed":
            metadata["status"] = "interrupted"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Resultados: {args.output_dir}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
