"""Treina a cabeça e depois cabeça + LoRA, validando apenas dentro de train."""

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
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dados
from modelo import MODEL, REVISION, collate, create, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=0, help="0 usa todo o treino disponível")
    parser.add_argument("--validation-limit", type=int, default=0, help="Somente para testes rápidos")
    parser.add_argument("--max-steps", type=int, default=0, help="Limite de atualizações por época; 0 sem limite")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir já existe; escolha outro diretório")
    if (min(args.head_epochs, args.lora_epochs, args.train_limit, args.validation_limit, args.max_steps) < 0
            or args.head_epochs + args.lora_epochs == 0
            or min(args.batch_size, args.gradient_accumulation, args.head_size, args.lora_rank, args.max_length) < 1
            or min(args.head_lr, args.lora_lr) <= 0 or not 0 < args.validation_fraction < 1):
        parser.error("Hiperparâmetros inválidos")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)

    dataset, classes, options = dados.load("train")  # Nunca carregar test durante o treino.
    train_ids, val_ids = train_test_split(
        list(range(len(dataset))), test_size=args.validation_fraction,
        random_state=args.seed, stratify=dataset["label"],
    )
    if args.train_limit:
        train_ids = random.Random(args.seed).sample(train_ids, min(args.train_limit, len(train_ids)))
    if args.validation_limit:
        val_ids = random.Random(args.seed).sample(val_ids, min(args.validation_limit, len(val_ids)))
    model = create(args.head_size, args.max_length)
    device = "cuda"

    def prepare(indices, shuffle_options):
        records = []
        rng = random.Random(args.seed)
        for index in indices:
            keys = list(options)
            if shuffle_options:
                rng.shuffle(keys)
            ordered = {k: options[k] for k in keys}
            sample = dataset[index]
            record = encode(model.tokenizer, sample["text"], dados.QUESTION, ordered, args.max_length)
            record["target"] = keys.index(classes[sample["label"]])
            records.append(record)
        return records

    train_records, val_records = prepare(train_ids, True), prepare(val_ids, False)

    def make_batch(records):
        batch = collate(records, model.tokenizer.pad_token_id, device)
        target = torch.tensor([r["target"] for r in records], device=device)
        return batch, target

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_records, batch_size=args.batch_size, shuffle=True,
                              generator=generator, collate_fn=make_batch)
    val_loader = DataLoader(val_records, batch_size=args.batch_size, collate_fn=make_batch)

    @torch.no_grad()
    def validate():
        model.eval()
        loss_sum = correct = count = 0
        for batch, target in val_loader:
            logits = model(batch)
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            correct += (logits.argmax(-1) == target).sum().item()
            count += len(target)
        return {"accuracy": correct / count, "loss": loss_sum / count}

    args.output_dir.mkdir(parents=True)
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": MODEL, "base_revision": REVISION, "dataset": dados.DATASET,
        "dataset_revision": dados.DATASET_REVISION, "training_split": "train",
        "train_indices": train_ids, "validation_indices": val_ids,
        "classes": classes, "question": dados.QUESTION, "options": options,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "python": sys.version.split()[0], "history": [],
        "packages": {n: importlib.metadata.version(n) for n in
                     ["torch", "transformers", "peft", "datasets", "huggingface_hub", "numpy", "scikit-learn"]},
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py")},
    }
    manifest_path = args.output_dir / "training.json"

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    write_manifest()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for phase, epochs in [("head", args.head_epochs), ("lora", args.lora_epochs)]:
            if not epochs:
                continue
            if phase == "lora":
                if (args.output_dir / "head.safetensors").exists():
                    model.head.load_state_dict(load_file(args.output_dir / "head.safetensors", device=device))
                model.add_lora(args.lora_rank)
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=args.head_lr if phase == "head" else args.lora_lr)
            print(f"{phase}: {sum(p.numel() for p in parameters):,} parâmetros treináveis; "
                  f"{len(train_ids)} treino, {len(val_ids)} validação.", flush=True)
            best = -1.0
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss = samples = steps = 0
                batches = len(train_loader)
                if args.max_steps:
                    batches = min(batches, args.max_steps * args.gradient_accumulation)
                for i, (batch, target) in enumerate(train_loader):
                    if i >= batches:
                        break
                    logits = model(batch)
                    loss = F.cross_entropy(logits, target, reduction="sum")
                    if not torch.isfinite(loss):
                        raise RuntimeError("Loss não finita")
                    loss.backward()
                    total_loss += loss.item()
                    samples += len(target)
                    if i % args.gradient_accumulation == 0:
                        accumulated_samples = 0
                    accumulated_samples += len(target)
                    if (i + 1) % args.gradient_accumulation == 0 or i + 1 == batches:
                        for parameter in parameters:
                            if parameter.grad is not None:
                                parameter.grad.div_(accumulated_samples)
                        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
                    if i == 0 or (i + 1) % 50 == 0 or i + 1 == batches:
                        print(f"{phase} época {epoch + 1}: {i + 1}/{batches} batches, "
                              f"loss={total_loss / samples:.4f}", flush=True)
                validation = validate()
                entry = {"phase": phase, "epoch": epoch + 1, "steps": steps,
                         "train_loss": total_loss / samples, "validation": validation}
                manifest["history"].append(entry)
                print(json.dumps(entry), flush=True)
                if validation["accuracy"] > best:
                    best = validation["accuracy"]
                    model.save(args.output_dir)
                    manifest["selected_checkpoint"] = entry
                write_manifest()
            del optimizer
        manifest["status"] = "completed"
    finally:
        if manifest["status"] != "completed":
            manifest["status"] = "interrupted"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest()
    print(f"Modelo salvo em {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
