from pathlib import Path
import argparse
import shutil

from PIL import Image
from tqdm import tqdm

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ANIMAL_KEYWORDS = {
    "dog", "cat", "lion", "tiger", "leopard", "cheetah", "jaguar",
    "pig", "hog", "boar", "cow", "ox", "bull", "buffalo", "bison",
    "horse", "zebra", "donkey", "sheep", "goat", "deer", "elephant",
    "bear", "wolf", "fox", "monkey", "ape", "gorilla", "chimpanzee",
    "panda", "koala", "kangaroo", "rabbit", "hare", "squirrel",
    "mouse", "rat", "hamster", "beaver", "otter", "raccoon",
    "bird", "eagle", "hawk", "owl", "duck", "goose", "swan",
    "chicken", "hen", "rooster", "turkey", "peacock", "parrot",
    "snake", "lizard", "crocodile", "alligator", "turtle", "frog",
    "fish", "shark", "whale", "dolphin", "seal", "penguin",
}

ANIMAL_KEYWORDS.update({
    "stork", "grouse", "partridge", "prairie chicken",
    "sturgeon", "dragonfly", "frog", "buffalo",
    "antelope", "gazelle", "ibex", "ram",
    "crane", "flamingo", "heron", "pelican",
    "bee", "beetle", "butterfly", "moth",
    "insect", "spider", "scorpion",
})

def looks_like_animal(label: str) -> bool:
    label = label.lower().replace("_", " ")
    return any(keyword in label for keyword in ANIMAL_KEYWORDS)

def safe_folder_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(",", "")
    )


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 1
    while True:
        candidate = path.parent / f"{path.stem}_{counter}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def ask_folder(prompt: str) -> Path:
    while True:
        value = input(prompt).strip().strip('"').strip("'")
        folder = Path(value)

        if folder.exists() and folder.is_dir():
            return folder

        print("That folder does not exist. Please paste a valid folder path.")


def ask_output_folder(prompt: str) -> Path:
    value = input(prompt).strip().strip('"').strip("'")
    folder = Path(value)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [y/N]: " if not default else " [Y/n]: "

    value = input(prompt + suffix).strip().lower()

    if not value:
        return default

    return value in {"y", "yes"}


# def load_model():
#    print("Loading AI model...")
#
#    weights = EfficientNet_B0_Weights.DEFAULT
#    model = efficientnet_b0(weights=weights)
#    model.eval()
#
#    preprocess = weights.transforms()
#    labels = weights.meta["categories"]
#
#    return model, preprocess, labels

def load_model():
    print("Loading Vision Transformer model...")

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    return model, preprocess, labels


def predict_image(image_path: Path, model, preprocess, labels):
    image = Image.open(image_path).convert("RGB")
    batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(batch).softmax(1)

    confidence, class_id = torch.max(prediction, dim=1)

    return labels[class_id.item()], confidence.item()


def sort_photos(input_dir: Path, output_dir: Path, confidence: float, move_files: bool):
    model, preprocess, labels = load_model()

    images = [
        path
        for path in input_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nFound {len(images)} images.")
    print("Sorting started...\n")

    for image_path in tqdm(images, desc="Sorting photos"):
        try:
            label, score = predict_image(image_path, model, preprocess, labels)

            if score < confidence:
                folder_name = "unknown_review"
                print(f"[UNKNOWN] {image_path.name} → {label} ({score:.2f})")

            elif not looks_like_animal(label):
                folder_name = "no_animal_review"
                print(f"[NO ANIMAL] {image_path.name} → {label} ({score:.2f})")

            else:
                folder_name = safe_folder_name(label)

            destination_folder = output_dir / folder_name
            destination_folder.mkdir(parents=True, exist_ok=True)

            destination = unique_destination(destination_folder / image_path.name)

            if move_files:
                shutil.move(str(image_path), str(destination))
            else:
                shutil.copy2(str(image_path), str(destination))

        except Exception as error:
            error_folder = output_dir / "errors"
            error_folder.mkdir(parents=True, exist_ok=True)
            print(f"Could not process {image_path}: {error}")

    print("\nDone.")
    print(f"Sorted photos saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sort animal photos into folders.")
    parser.add_argument("--input", help="Folder containing unsorted photos")
    parser.add_argument("--output", help="Folder where sorted photos will be stored")
    #parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")

    args = parser.parse_args()

    print("\nAnimal Photo Sorter")
    print("===================")

    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = ask_folder("\nPaste the folder path where all images are stored: ")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = ask_output_folder("Paste the folder path where sorted images should be saved: ")

    if args.move:
        move_files = True
    else:
        move_files = ask_yes_no("Do you want to move files instead of copying them?", default=False)

    sort_photos(
        input_dir=input_dir,
        output_dir=output_dir,
        confidence=args.confidence,
        move_files=move_files,
    )


if __name__ == "__main__":
    main()
