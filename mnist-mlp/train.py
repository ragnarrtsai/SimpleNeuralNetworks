import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

TOPIC_DIR = Path(__file__).parent
INPUT_DIR = TOPIC_DIR / "input"
MODEL_DIR = TOPIC_DIR / "model"
OUTPUT_DIR = TOPIC_DIR / "output"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    run_output_dir = OUTPUT_DIR / RUN_ID
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"run id: {RUN_ID}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(INPUT_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(INPUT_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        test_acc = correct / len(test_loader.dataset)

        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "test_acc": test_acc})

    model_path = MODEL_DIR / f"{RUN_ID}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"saved model → {model_path}")

    history_path = run_output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"saved training history → {history_path}")
    print(f"to release this run, copy {model_path.name} to the topic root (e.g. as latest.pt)")


if __name__ == "__main__":
    main()
