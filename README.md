# Auto Multi-Crop GC: Sub-pixel Edge Fitting & Web UI

Auto Multi-Crop GC is a Python application for auto-segmenting and cropping individual photos from multi-photo scan sheets. It features RANSAC sub-pixel edge detection, perspective correction, quality-control flagging, and an interactive HTML5 Web UI for visual inspection and manual fine-tuning.

---

## ✨ Features

- **Automatic Multi-Photo Detection**: Automatically detects multiple photos per scanned sheet using morphological segmentation and contour analysis.
- **RANSAC Sub-Pixel Edge Refinement**: Uses Sobel gradient magnitude rays and RANSAC outlier-resistant 2D line fitting to tightly snap crop boundaries to real photo edges while ignoring scanbed background artifacts.
- **Perspective Transform & Edge Preservation**: Flattens non-rectangular or angled scans via 4-point homography warping, with configurable margin expansion (`--margin`) to preserve 100% of photo edges.
- **EfficientNetV2 AI Auto-Rotation Engine ("Up is Up")**: Uses a 98.13% accurate EfficientNetV2 ONNX vision model specifically trained for 4-way 90° image orientation classification to automatically rotate cropped photos upright.
- **Quality Control & Flagging Engine**: Automatically detects non-orthogonal corners or extreme skewing and flags problematic crops for review.
- **Interactive Dual-Tab Web Application**:
  - **Tab 1 (📐 1. Crop Quad Editor)**: Pan/zoom canvas to inspect full scan beds, drag corner handles, add/delete crops, save JSON, export debug overlays, and re-crop photos.
  - **Tab 2 (🖼️ 2. Cropped Photo Gallery)**: Dedicated orientation & rotation workspace displaying a responsive thumbnail grid of all cropped photos with instant, zero-latency in-browser CSS rotation controls (`↺ 90° CCW` and `↻ 90° CW`), plus a **💾 Apply & Save Rotations** batch button and **🤖 Auto-Rotate All** AI button.

---

## 🛠️ Installation & Requirements

Requires **Python 3.8+** and standard data science / computer vision libraries:

```bash
pip install opencv-python numpy
```

---

## 📂 Project Structure

```text
VA_scans/
├── ScanOldPhotos/         # Input raw scan files (.jpg, .jpeg, .png)
├── ScanOldPhotosCropped/  # Exported individual cropped photos
├── ScanOldPhotosDebug/    # Generated visual debug overlays
├── crops_manifest.json    # JSON manifest containing detected quad coordinates
├── orientation_duarte.onnx# EfficientNetV2 ONNX model (auto-downloaded on first run)
├── photo_cropper.py       # Core CLI script & web server backend
├── preview.html           # Interactive HTML5 Web UI frontend
└── LICENSE                # CC BY-NC-SA 4.0 License & Third-Party Attributions
```

---

## 🚀 CLI Usage

Run the core script using python:

```bash
# Run the complete pipeline (Detect -> Crop -> Auto-Rotate -> Debug -> Web Server)
python photo_cropper.py --all

# Run individual pipeline stages
python photo_cropper.py --detect       # Detect photos & generate crops_manifest.json
python photo_cropper.py --crop         # Export cropped photos to ScanOldPhotosCropped/
python photo_cropper.py --auto-rotate  # Auto-detect orientation & rotate cropped photos
python photo_cropper.py --debug        # Generate visual debug images in ScanOldPhotosDebug/
python photo_cropper.py --server       # Launch Web UI on http://localhost:8000

# Custom Options
python photo_cropper.py --server --port 8080   # Custom HTTP server port
python photo_cropper.py --crop --margin 4        # Set crop edge expansion margin in pixels
```

---

## 🌐 Web Editor Interface

Launch the server with `python photo_cropper.py --server` and navigate to `http://localhost:8000` in your browser.

### Controls & Navigation
- **Pan Viewport**: Click and drag on empty canvas area.
- **Zoom**: Mouse wheel or `🔍+` / `🔍-` toolbar buttons.
- **Corner Adjustment**: Click and drag any white corner handle on a selected photo quad.
- **Undo Edit**: Press `Ctrl+Z` (or `Cmd+Z` on Mac) or click **↩️ Undo**.
- **Save JSON Only**: Click **💾 Save JSON Only** to save corner modifications to `crops_manifest.json`.
- **Export Debug Overlays**: Click **🖼️ Export Debug Overlays** to generate annotated debug images.
- **Save & Re-Crop All**: Click **✂️ Save & Re-Crop All** to export high-quality cropped JPEG images.
- **Gallery Mode & Batch Save**: Click `↺` or `↻` on any thumbnail for instant CSS rotation; click **💾 Apply & Save Rotations** to persist all rotations to disk in one batch.

---

## 📄 License & Usage Rights

This project is licensed under the **[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](LICENSE)**.

- **Non-Commercial**: Allowed for personal, academic, educational, and open non-commercial use with attribution.
- **No Commercial Use**: Commercial use of any sort (in original or modified form) is strictly prohibited.
- **ShareAlike**: Derivative works must be distributed under the exact same CC BY-NC-SA 4.0 license.
- **Third-Party Attribution**: Includes DuarteBarbosa EfficientNetV2 model weights (MIT License). See [LICENSE](LICENSE) for full legal text and third-party notices.
