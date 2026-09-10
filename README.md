# Auto Cropper GC 🖼️✂️

An automatic, batch-able photo scan segmenter, sub-pixel RANSAC crop detector, and perspective corrector with an interactive HTML5 preview editor.

Designed to automatically process high-resolution flatbed photo scans containing 1–4 physical prints, detect precise straight edges with sub-pixel RANSAC line fitting, correct sub-90° tilt/rotations, and export individual perspective-corrected JPEGs.

---

## 🌟 Key Features

1. **Automatic Batch Segmentation**:
   - Automatically segments individual photographs from high-resolution scanner scans.
   - Handles tight photo spacing, thin white margins, and light background reflections.

2. **Sub-Pixel RANSAC Line Fitting**:
   - Uses directional Sobel gradient peak searching paired with **RANSAC outlier rejection** (`fit_line_ransac()`) to fit sub-pixel accurate straight lines along outer paper borders.
   - Completely ignores internal photo texture noise, clothing lines, and neighboring photo borders.

3. **Sub-90° Rotation & Perspective Warp**:
   - Calculates 4-point perspective warp matrices (`cv2.getPerspectiveTransform`) to straighten tilted or crooked scanned prints.

4. **Scanner Bed Artifact Exclusion**:
   - Ignores outer perimeter glass/frame dark shadow artifacts (`SCANNER_BORDER_MARGIN = 15px`).

5. **Interactive HTML5 Web Editor (`preview.html`)**:
   - Built-in lightweight web server (`photo_cropper.py --server`).
   - Interactive canvas displaying overlayed 4-corner crop quadrilaterals.
   - **Draggable Corner Handles**: Drag any corner point to fine-tune crop boundaries.
   - **Undo Support (`Ctrl+Z`)**: Multi-level state history tracking for corner edits, additions, and deletions.
   - **Save JSON Only**: Instantly save crop manifest coordinates without full image re-cropping.
   - **Export Debug Overlays**: Export visual debug overlay images showing edge sampling points.

---

## 📁 Directory Structure

```text
VA_scans/
├── ScanOldPhotos/         # Source scan JPEGs
├── ScanOldPhotosCropped/  # Exported individual cropped JPEGs
├── ScanOldPhotosDebug/    # Visual debug overlay images
├── crops_manifest.json    # JSON manifest containing 4-corner quad coordinates
├── photo_cropper.py       # Core Python engine and Web API server
├── preview.html           # HTML5 Canvas web editor
└── README.md
```

---

## 🚀 Getting Started

### Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Pillow

### Usage Commands

```bash
# 1. Run full pipeline: Detect -> Crop -> Debug Overlays -> Launch Web Server
python3 photo_cropper.py --all

# 2. Run detection and generate crops_manifest.json
python3 photo_cropper.py --detect

# 3. Export cropped photos from manifest (with optional margin padding)
python3 photo_cropper.py --crop --margin 3

# 4. Generate visual debug overlay images
python3 photo_cropper.py --debug

# 5. Start Preview Web Server only
python3 photo_cropper.py --server
```

---

## 🌐 Web Preview Editor

Access the web interface at **`http://localhost:8000`** while `--server` is running.

- **Pan & Zoom**: Click and drag background or use scroll wheel.
- **Adjust Corners**: Drag blue handle points on any photo.
- **Undo Edit**: Click "Undo" or press `Ctrl+Z`.
- **Add / Delete Crop**: Add new photo crops or remove false positives.
- **Save & Re-Crop**: Update JSON manifest and re-export cropped JPEGs.
