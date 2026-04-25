# 🐾 Animal Photo Sorter

Automatically sort thousands of animal photos into folders using AI.

This tool scans your images, recognize the animal, and organizes them into folders like:

```
sorted_photos/
  lion/
  leopard/
  pig/
  elephant/
  unknown_review/
  errors/
```

---

## 🚀 Features

* Works with thousands of images (tested on large datasets)
* Uses AI (Vision Transformer model) for classification
* Interactive (no technical knowledge required)
* Automatically creates folders
* Handles duplicate filenames safely
* Separates low-confidence images for review

---

## 📦 Installation (Super Simple)

First install:

👉 uv

Then run:

```bash
uv sync
```

That’s it. Everything installs automatically.

---

## ▶️ How to Run

```bash
uv run sort-photos
```

The program will ask you:

1. 📁 Where your images are stored
2. 📂 Where to save sorted images
3. 🔁 Whether to copy or move files

Just paste the folder paths and press Enter.

---

## 🧠 How It Works

* Uses a pretrained AI model (EfficientNet)
* Predicts the animal in each image
* Moves/copies the image into a matching folder

If the model is unsure:

```
→ unknown_review/
```

If something fails:

```
→ errors/
```

---

## ⚙️ Supported Image Formats

* `.jpg`
* `.jpeg`
* `.png`
* `.webp`
* `.bmp`

---

## 🧪 Tips for Best Results

* Start with a small folder (50–200 images) to test
* Use **copy mode first** (safer)
* Increase confidence threshold if results are messy
* Expect ~5–15% images in `unknown_review`

---

## 🛠️ Tech Stack

* Python
* PyTorch
* TorchVision
* EfficientNet model

---
