import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn

ATTRS = ["Attr1", "Attr2", "Attr3", "Attr4"]

def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 4),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load("models/checkpoint.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess(img_path):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return t(img).unsqueeze(0)

def infer(img_path):
    model = load_model()
    x = preprocess(img_path)

    with torch.no_grad():
        pred = model(x).squeeze()

    results = [ATTRS[i] for i, p in enumerate(pred) if p >= 0.5]
    print("Predicted attributes:", results)

# ------------- CLI Support --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    infer(args.img)