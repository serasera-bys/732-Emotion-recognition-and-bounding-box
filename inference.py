from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image
import torch

from models import build_model
from transforms import build_image_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image inference from a saved checkpoint.")
    parser.add_argument("--image", default="test_images/surprise/ffhq_24.png")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=1)
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inferencer:
    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        self.device = select_device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint["class_names"])
        self.model = build_model(checkpoint["arch"], num_classes=len(self.class_names), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = build_image_transform(int(checkpoint["image_size"]), train=False)

    def preprocess_image(self, image):
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)

    def predict(self, image):
        tensor = self.preprocess_image(image)
        return self.predict_tensor(tensor)

    def predict_tensor(self, tensor: torch.Tensor):
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class].item())
        return predicted_class, confidence

    def predict_distribution_tensor(self, tensor: torch.Tensor, top_k: int = 3):
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_class].item())
            k = max(1, min(top_k, probabilities.numel()))
            values, indices = torch.topk(probabilities, k=k)
        score_map = {
            self.class_names[index]: float(probabilities[index].item())
            for index in range(probabilities.numel())
        }
        top_predictions = [
            {
                "class_name": self.class_names[int(index.item())],
                "confidence": float(value.item()),
            }
            for value, index in zip(values, indices)
        ]
        return predicted_class, confidence, score_map, top_predictions

    def predict_top_k(self, image, top_k: int = 3):
        tensor = self.preprocess_image(image)
        return self.predict_top_k_tensor(tensor, top_k=top_k)

    def predict_top_k_tensor(self, tensor: torch.Tensor, top_k: int = 3):
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            k = max(1, min(top_k, probabilities.numel()))
            values, indices = torch.topk(probabilities, k=k)
        return [
            {
                "class_name": self.class_names[int(index.item())],
                "confidence": float(value.item()),
            }
            for value, index in zip(values, indices)
        ]

    def predict_with_labels(self, image, top_k: int = 1):
        predicted_idx, confidence = self.predict(image)
        payload = {
            "predicted_class": self.class_names[predicted_idx],
            "confidence": confidence,
        }
        if top_k > 1:
            payload["top_k_predictions"] = self.predict_top_k(image, top_k=top_k)
        return payload


def main() -> None:
    args = parse_args()
    inferencer = Inferencer(args.checkpoint, device=args.device)
    print(json.dumps(inferencer.predict_with_labels(args.image, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
