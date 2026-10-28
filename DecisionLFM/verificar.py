"""Recarrega o checkpoint e confere a validação registrada e uma passagem por decisão."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

import dados
from modelo import DecisionLFM, collate, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.checkpoint / "training.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Treinamento incompleto")
    if set(manifest["train_indices"]) & set(manifest["validation_indices"]):
        raise ValueError("Sobreposição entre treino e validação")
    torch.set_num_threads(8)
    model = DecisionLFM.load(args.checkpoint)
    dataset, classes, options = dados.load("train")
    indices = manifest["validation_indices"]
    size = manifest["args"]["batch_size"]
    correct = count = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(indices), size):
            selected = indices[start:start + size]
            records = [encode(model.tokenizer, dataset[i]["text"], dados.QUESTION, options,
                              model.max_length) for i in selected]
            targets = torch.tensor([dataset[i]["label"] for i in selected], device="cuda")
            logits = model(collate(records, model.tokenizer.pad_token_id, "cuda"))
            correct += (logits.argmax(-1) == targets).sum().item()
            count += len(selected)
            loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
    accuracy, loss = correct / count, loss_sum / count
    expected = manifest["selected_checkpoint"]["validation"]
    if accuracy != expected["accuracy"] or abs(loss - expected["loss"]) > 1e-4:
        raise ValueError(f"Validação após recarregar divergiu: {accuracy=}, {loss=}, {expected=}")

    calls = []
    hook = model.backbone.register_forward_hook(lambda *unused: calls.append(1))
    result = model.decidir(dataset[indices[0]]["text"], dados.QUESTION, options)
    hook.remove()
    if len(calls) != 1 or len(result["probabilities"]) != len(classes):
        raise ValueError("A decisão deve usar uma passagem e pontuar todas as alternativas")
    two = model.decidir("My card payment was declined.", "Which banking intent?",
                       {"card": "Card payment declined", "cash": "Cash withdrawal problem"})
    if set(two["probabilities"]) != {"card", "cash"}:
        raise ValueError("O checkpoint deve aceitar um novo esquema de alternativas")
    print(json.dumps({"status": "passed", "validation_records": count,
                      "validation_accuracy": accuracy, "validation_loss": loss,
                      "backbone_calls_per_decision": len(calls),
                      "dynamic_options": [len(classes), len(two["probabilities"])]}, indent=2))


if __name__ == "__main__":
    main()
