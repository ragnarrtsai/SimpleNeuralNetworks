import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP

TOPIC_DIR = Path(__file__).parent
INPUT_DIR = TOPIC_DIR / "input"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(TOPIC_DIR / "latest.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(INPUT_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = MLP().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"weights: {args.weights}")
    print(f"test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
