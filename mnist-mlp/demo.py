import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

TOPIC_DIR = Path(__file__).parent
INPUT_DIR = TOPIC_DIR / "input"
OUTPUT_DIR = TOPIC_DIR / "output"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to a .pt file")
    parser.add_argument("--run-id", required=True, help="output subfolder name (e.g. yyyyMMdd_HHmmss of the training run)")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--timing-iters", type=int, default=1000)
    args = parser.parse_args()

    showcase_dir = OUTPUT_DIR / args.run_id / "showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(INPUT_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256)

    device = torch.device(args.device)
    model = MLP().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # 1. Sample predictions grid
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_ds), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    for ax, idx in zip(axes.flat, sample_idx):
        x, y = test_ds[idx]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).argmax(1).item()
        img = x.squeeze().numpy() * 0.3081 + 0.1307
        ax.imshow(img, cmap="gray")
        color = "green" if pred == y else "red"
        ax.set_title(f"pred: {pred}  /  true: {y}", color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample predictions on MNIST test set")
    fig.tight_layout()
    fig.savefig(showcase_dir / "samples.png", dpi=130)
    plt.close(fig)

    # 2. Training curve
    history_path = OUTPUT_DIR / args.run_id / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        epochs = [h["epoch"] for h in history]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(epochs, [h["train_loss"] for h in history], "b-o", label="train_loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax2 = ax1.twinx()
        ax2.plot(epochs, [h["test_acc"] for h in history], "r-s", label="test_acc")
        ax2.set_ylabel("test acc", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        fig.suptitle("Training curve")
        fig.tight_layout()
        fig.savefig(showcase_dir / "training_curve.png", dpi=130)
        plt.close(fig)

    # 3. Confusion matrix on full test set
    confusion = np.zeros((10, 10), dtype=int)
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                confusion[t, p] += 1
            correct += (preds == y).sum().item()
    test_acc = correct / len(test_ds)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(confusion, cmap="Blues")
    threshold = confusion.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > threshold else "black",
                    fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix  (test acc: {test_acc:.2%})")
    fig.tight_layout()
    fig.savefig(showcase_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # 4. Single-sample inference timing
    sample_x = test_ds[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(20):
            model(sample_x)
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.timing_iters):
            model(sample_x)
    sync(device)
    avg_us = (time.perf_counter() - start) / args.timing_iters * 1e6

    metrics = {
        "test_accuracy": test_acc,
        "single_sample_inference_us": avg_us,
        "device": device.type,
        "timing_iters": args.timing_iters,
        "weights": str(args.weights),
        "confusion_matrix": confusion.tolist(),
    }
    (showcase_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"=== Showcase ({device.type}) ===")
    print(f"  test accuracy:               {test_acc:.4f}")
    print(f"  single-sample inference:     {avg_us:.1f} µs")
    print(f"  saved → {showcase_dir}")


if __name__ == "__main__":
    main()
