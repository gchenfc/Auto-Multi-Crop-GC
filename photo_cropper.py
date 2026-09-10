#!/usr/bin/env python3
"""
Photo Cropper & Crop Preview Server
Automatically detects 2-4 individual photos from scanned JPG images, performs RANSAC sub-pixel 
outermost edge fitting and perspective correction, exports individual photos, and hosts an 
interactive web UI for previewing and editing crop polygons.
"""

import os
import sys
import json
import traceback
import glob
import math
import argparse
import http.server
import socketserver
import urllib.parse
import cv2
import numpy as np
import threading

MODEL_LOCK = threading.Lock()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default Directory Configuration
INPUT_DIR = os.path.join(SCRIPT_DIR, "ScanOldPhotos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ScanOldPhotosCropped")
DEBUG_DIR = os.path.join(SCRIPT_DIR, "ScanOldPhotosDebug")
MANIFEST_FILE = os.path.join(SCRIPT_DIR, "crops_manifest.json")
PREVIEW_HTML_FILE = os.path.join(SCRIPT_DIR, "preview.html")
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "mobilenetv2-12.onnx")
PORT = 8000
DEFAULT_MARGIN = 2  # Expand crop quad by N pixels outward to preserve 100% of edges
SCANNER_BORDER_MARGIN = 15  # Ignore outer N pixels of scan bed to eliminate scanner glass/frame artifacts


def order_corners(pts):
    """
    Orders 4 corners as: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def expand_quad_corners(corners, margin=DEFAULT_MARGIN):
    """
    Slightly expands a 4-corner polygon outward from its centroid by 'margin' pixels.
    Ensures no border or corner slivers are cut off.
    """
    pts = order_corners(corners)
    if margin == 0:
        return pts
        
    center = pts.mean(axis=0)
    expanded = []
    for pt in pts:
        vec = pt - center
        dist = np.linalg.norm(vec)
        if dist > 1e-5:
            new_pt = pt + (vec / dist) * margin
            expanded.append(new_pt)
        else:
            expanded.append(pt)
    return np.array(expanded, dtype=np.float32)


def fit_line_ransac(points, inlier_threshold=2.0, max_iterations=500):
    """
    Fits a 2D line (vx, vy, x0, y0) through points using RANSAC outlier rejection.
    Ignores texture noise, interior lines, and adjacent photo points.
    """
    if len(points) < 4:
        if len(points) >= 2:
            pts = np.array(points, dtype=np.float32)
            line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            return line.flatten()
        return None

    pts = np.array(points, dtype=np.float32)
    N = len(pts)
    best_inliers = []
    best_count = -1

    for _ in range(max_iterations):
        idx = np.random.choice(N, 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        
        vec = p2 - p1
        norm_v = np.linalg.norm(vec)
        if norm_v < 1e-5:
            continue
        
        A = -vec[1] / norm_v
        B = vec[0] / norm_v
        C = -(A * p1[0] + B * p1[1])
        
        dists = np.abs(A * pts[:, 0] + B * pts[:, 1] + C)
        inliers = pts[dists <= inlier_threshold]
        
        if len(inliers) > best_count:
            best_count = len(inliers)
            best_inliers = inliers

    if len(best_inliers) >= 4:
        line = cv2.fitLine(best_inliers, cv2.DIST_L2, 0, 0.01, 0.01)
        return line.flatten()
    else:
        line = cv2.fitLine(pts, cv2.DIST_L1, 0, 0.01, 0.01)
        return line.flatten()


def refine_quad_ransac(gray_img, approx_quad, search_radius=12, border_margin=10):
    """
    Refines a 4-corner polygon using RANSAC sub-pixel line fitting.
    Searches along normal rays for Sobel gradient peaks and applies RANSAC outlier rejection.
    """
    ordered = order_corners(approx_quad)
    h, w = gray_img.shape[:2]
    
    # Clamp initial points to stay within scanbed area
    ordered[:, 0] = np.clip(ordered[:, 0], border_margin, w - 1 - border_margin)
    ordered[:, 1] = np.clip(ordered[:, 1], border_margin, h - 1 - border_margin)
    
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    center = ordered.mean(axis=0)
    lines = []
    sampled_edge_points = []
    
    for i in range(4):
        p1 = ordered[i]
        p2 = ordered[(i+1)%4]
        seg_length = np.linalg.norm(p2 - p1)
        num_samples = max(15, int(seg_length / 4.0))
        
        t_vals = np.linspace(0.05, 0.95, num_samples)
        sample_pts = p1 + np.outer(t_vals, (p2 - p1))
        vec = p2 - p1
        
        norm_v = np.array([-vec[1], vec[0]])
        norm_v = norm_v / max(1e-5, np.linalg.norm(norm_v))
        mid_seg = (p1 + p2) / 2.0
        if np.dot(norm_v, mid_seg - center) < 0:
            norm_v = -norm_v
            
        edge_pts = []
        for sp in sample_pts:
            best_pt = None
            best_mag = -1.0
            
            for offset in range(-search_radius, search_radius + 1):
                ep = sp + norm_v * offset
                ex, ey = int(round(ep[0])), int(round(ep[1]))
                if border_margin <= ex < w - border_margin and border_margin <= ey < h - border_margin:
                    m = mag[ey, ex]
                    g = gray_img[ey, ex]
                    if m > 25.0 and g > 110:
                        if m > best_mag:
                            best_mag = m
                            best_pt = [ex, ey]
                            
            if best_pt is not None:
                edge_pts.append(best_pt)
                sampled_edge_points.append(best_pt)
                
        line_params = fit_line_ransac(edge_pts, inlier_threshold=2.0)
        if line_params is not None:
            lines.append(line_params)
        else:
            lines.append((vec[0], vec[1], p1[0], p1[1]))
            
    refined_corners = []
    for i in range(4):
        vx1, vy1, x1, y1 = lines[i]
        vx2, vy2, x2, y2 = lines[(i-1)%4]
        A1, B1, C1 = vy1, -vx1, vy1*x1 - vx1*y1
        A2, B2, C2 = vy2, -vx2, vy2*x2 - vx2*y2
        det = A1*B2 - A2*B1
        if abs(det) > 1e-3:
            ix = (C1*B2 - C2*B1) / det
            iy = (A1*C2 - A2*C1) / det
            dist = np.hypot(ix - ordered[i][0], iy - ordered[i][1])
            # Strict displacement limit: max 18px displacement from initial minAreaRect corner
            if dist < 18.0:
                refined_corners.append([ix, iy])
            else:
                refined_corners.append(ordered[i].tolist())
        else:
            refined_corners.append(ordered[i].tolist())
            
    return np.array(refined_corners, dtype=np.float32), sampled_edge_points


def detect_photos_in_image(filepath):
    """
    Detects individual photo quadrilaterals in a scan file using RANSAC sub-pixel line fitting.
    """
    img = cv2.imread(filepath)
    if img is None:
        return [], (0, 0), []
    h, w = img.shape[:2]
    
    max_dim = 1200.0
    scale = max_dim / max(h, w)
    small = cv2.resize(img, (0,0), fx=scale, fy=scale)
    sh, sw = small.shape[:2]
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    _, thresh = cv2.threshold(blur, 232, 255, cv2.THRESH_BINARY_INV)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_clean)
    
    bm_small = max(2, int(round(SCANNER_BORDER_MARGIN * scale)))
    thresh_clean[:, :bm_small] = 0
    thresh_clean[:, -bm_small:] = 0
    thresh_clean[:bm_small, :] = 0
    thresh_clean[-bm_small:, :] = 0
    
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = sh * sw * 0.02
    raw_cnts = [c for c in contours if cv2.contourArea(c) > min_area]
    
    scale_mid = 2000.0 / max(h, w)
    mid = cv2.resize(img, (0,0), fx=scale_mid, fy=scale_mid)
    gray_mid = cv2.cvtColor(mid, cv2.COLOR_BGR2GRAY)
    bm_mid = max(5, int(round(SCANNER_BORDER_MARGIN * scale_mid)))
    
    photo_quads = []
    all_debug_points = []
    
    for c in raw_cnts:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        solidity = area / max(1.0, cv2.contourArea(hull))
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        
        boxes_to_refine = []
        if (solidity < 0.88 and aspect > 1.8) or area > (sh * sw * 0.28):
            mask = np.zeros_like(thresh_clean)
            cv2.drawContours(mask, [c], -1, 255, -1)
            split_found = False
            for k_size in [15, 21, 27, 35]:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
                eroded = cv2.erode(mask, kernel, iterations=2)
                sub_cnts, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sub_valid = [sc for sc in sub_cnts if cv2.contourArea(sc) > (sh * sw * 0.01)]
                if len(sub_valid) > 1:
                    split_found = True
                    for sc in sub_valid:
                        sub_mask = np.zeros_like(thresh_clean)
                        cv2.drawContours(sub_mask, [sc], -1, 255, -1)
                        sub_restored = cv2.dilate(sub_mask, kernel, iterations=2)
                        rcnts, _ = cv2.findContours(sub_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if rcnts:
                            boxes_to_refine.append(cv2.boxPoints(cv2.minAreaRect(rcnts[0])))
                    break
            if not split_found:
                boxes_to_refine.append(cv2.boxPoints(rect))
        else:
            boxes_to_refine.append(cv2.boxPoints(rect))
            
        for approx_box in boxes_to_refine:
            approx_mid = (approx_box / scale) * scale_mid
            ref_mid, edge_pts_mid = refine_quad_ransac(gray_mid, approx_mid, search_radius=12, border_margin=bm_mid)
            ref_orig = ref_mid / scale_mid
            
            expanded_box = expand_quad_corners(ref_orig, margin=DEFAULT_MARGIN)
            photo_quads.append(expanded_box)
            all_debug_points.extend([ [pt[0]/scale_mid, pt[1]/scale_mid] for pt in edge_pts_mid ])
            
    return photo_quads, (w, h), all_debug_points


def build_manifest(input_dir=INPUT_DIR, manifest_path=MANIFEST_FILE):
    """
    Scans input directory, detects photos across all scans using RANSAC refinement,
    checks tolerances, and builds crops_manifest.json.
    """
    scans = sorted(glob.glob(os.path.join(input_dir, "*.jpeg"))) + \
            sorted(glob.glob(os.path.join(input_dir, "*.jpg"))) + \
            sorted(glob.glob(os.path.join(input_dir, "*.png")))
            
    print(f"Detecting photo crops across {len(scans)} scan files in '{input_dir}'...")
    manifest = {"scans": {}}
    
    for s in scans:
        name = os.path.basename(s)
        quads, (w, h), _ = detect_photos_in_image(s)
        photos = []
        
        for idx, q in enumerate(quads, start=1):
            corners = np.round(q, 2).tolist()
            corners_ord = order_corners(corners).tolist()
            
            flagged = False
            reasons = []
            
            vecs = []
            for i in range(4):
                v = np.array(corners_ord[(i+1)%4]) - np.array(corners_ord[i])
                vecs.append(v / max(1e-5, np.linalg.norm(v)))
            
            for i in range(4):
                v1 = vecs[i]
                v2 = -vecs[(i-1)%4]
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                ang = np.degrees(np.arccos(dot))
                if abs(ang - 90.0) > 10.0:
                    flagged = True
                    reasons.append(f"Corner {i+1} angle ({ang:.1f}°) deviates from 90°")
                    
            photos.append({
                "id": f"{os.path.splitext(name)[0]}_photo_{idx}",
                "corners": corners_ord,
                "flagged": flagged,
                "flag_reasons": list(set(reasons))
            })
            
        manifest["scans"][name] = {
            "width": w,
            "height": h,
            "photo_count": len(photos),
            "photos": photos
        }
        print(f"  [{name}] Found {len(photos)} photos (Flagged: {sum(1 for p in photos if p['flagged'])})")
        
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to '{manifest_path}'.")
    return manifest


def crop_image_quad(img, corners, margin=0):
    """
    Performs 4-point perspective warp on img given 4 corners [[x0,y0],[x1,y1],[x2,y2],[x3,y3]].
    """
    pts = order_corners(corners)
    if margin > 0:
        pts = expand_quad_corners(pts, margin=margin)
        
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(round(widthA)), int(round(widthB)))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(round(heightA)), int(round(heightB)))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
    return warped


def process_single_scan_crop(item):
    name, scan_info, input_dir, output_dir, margin = item
    scan_path = os.path.join(input_dir, name)
    if not os.path.exists(scan_path):
        return 0
    img = cv2.imread(scan_path)
    if img is None:
        return 0
    
    count = 0
    for photo in scan_info["photos"]:
        corners = photo["corners"]
        pid = photo["id"]
        cropped = crop_image_quad(img, corners, margin=margin)
        out_path = os.path.join(output_dir, f"{pid}.jpg")
        cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        count += 1
    return count


def run_crop_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, manifest_path=MANIFEST_FILE, margin=0):
    """
    Reads manifest JSON and exports all perspective-corrected photo crops in parallel.
    """
    import concurrent.futures
    if not os.path.exists(manifest_path):
        build_manifest(input_dir, manifest_path)
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cropping photos into '{output_dir}'...")
    
    items = [(name, scan_info, input_dir, output_dir, margin) for name, scan_info in manifest["scans"].items()]
    total_cropped = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        results = executor.map(process_single_scan_crop, items)
        total_cropped = sum(results)
            
    print(f"Successfully exported {total_cropped} cropped photos to '{output_dir}'.")


def process_single_debug_image(item):
    name, scan_info, input_dir, debug_dir = item
    scan_path = os.path.join(input_dir, name)
    if not os.path.exists(scan_path):
        return
    img = cv2.imread(scan_path)
    if img is None:
        return
        
    h, w = img.shape[:2]
    scale = 1200.0 / max(h, w)
    small = cv2.resize(img, (0,0), fx=scale, fy=scale)
    
    colors = [(0, 255, 0), (255, 165, 0), (0, 255, 255), (255, 0, 255)]
    quads, _, debug_pts = detect_photos_in_image(scan_path)
    
    # Draw edge points
    for pt in debug_pts:
        px, py = int(round(pt[0] * scale)), int(round(pt[1] * scale))
        cv2.circle(small, (px, py), 1, (0, 0, 255), -1)
        
    # Draw final quads with 1px thickness
    for idx, photo in enumerate(scan_info["photos"]):
        corners = (np.array(photo["corners"]) * scale).astype(np.int32)
        col = colors[idx % len(colors)]
        cv2.polylines(small, [corners], isClosed=True, color=col, thickness=1)
        cv2.putText(small, f"P{idx+1}", (corners[0][0]+5, corners[0][1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
    out_name = f"debug_{os.path.splitext(name)[0]}.jpg"
    cv2.imwrite(os.path.join(debug_dir, out_name), small)


def generate_debug_images(input_dir=INPUT_DIR, debug_dir=DEBUG_DIR, manifest_path=MANIFEST_FILE):
    """
    Generates visual debug overlay images showing detected crops and gradient edge points in parallel.
    """
    import concurrent.futures
    if not os.path.exists(manifest_path):
        build_manifest(input_dir, manifest_path)
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Generating visual debug overlays in '{debug_dir}'...")
    
    items = [(name, scan_info, input_dir, debug_dir) for name, scan_info in manifest["scans"].items()]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        list(executor.map(process_single_debug_image, items))
        
    print(f"Saved debug images for all scans in '{debug_dir}'.")


ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "orientation_duarte.onnx")
DUARTE_ONNX_URL = "https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection/resolve/main/orientation_model_v2_0.9882.onnx"

def load_orientation_models():
    """
    Loads DuarteBarbosa EfficientNetV2 ONNX orientation classifier model (98.13% ground truth accuracy).
    Auto-downloads weights on first use if missing.
    """
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Downloading EfficientNetV2 orientation model weights to '{ONNX_MODEL_PATH}'...", flush=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(DUARTE_ONNX_URL, ONNX_MODEL_PATH)
            print("Download complete.", flush=True)
        except Exception as e:
            print(f"Warning: Failed to download orientation model weights: {e}")
            return None

    net = None
    if os.path.exists(ONNX_MODEL_PATH):
        try:
            net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
        except Exception as e:
            print(f"Warning: Failed to load ONNX model '{ONNX_MODEL_PATH}': {e}")
            
    return net


def rotate_cv2_image(img, angle_cw):
    """
    Rotates image by angle_cw degrees (0, 90, 180, 270) clockwise.
    """
    angle_cw = int(angle_cw) % 360
    if angle_cw == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle_cw == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle_cw == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def classify_image_orientation(img, net=None):
    """
    Classifies image orientation (0, 90, 180, 270 CW) using EfficientNetV2 98.13% accurate model.
    """
    if net is None:
        net = load_orientation_models()
        
    if net is None:
        return 0, "default", 1.0
        
    class_to_angle = {0: 0, 1: 90, 2: 180, 3: 270}
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    with MODEL_LOCK:
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0, size=(224, 224),
                                     mean=(0.485*255, 0.456*255, 0.406*255),
                                     swapRB=True, crop=False) / std
        net.setInput(blob)
        out = net.forward()[0]
        
    exp_out = np.exp(out - np.max(out))
    probs = exp_out / np.sum(exp_out)
    pred_cls = int(np.argmax(probs))
    pred_angle = class_to_angle[pred_cls]
    confidence = float(probs[pred_cls])

    return pred_angle, "efficientnet_v2", confidence


def process_single_auto_rotate(item):
    pid, crop_path, net = item
    if not os.path.exists(crop_path):
        return None
    img = cv2.imread(crop_path)
    if img is None:
        return None
    angle, method, score = classify_image_orientation(img, net)
    if angle != 0:
        rotated_img = rotate_cv2_image(img, angle)
        cv2.imwrite(crop_path, rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"  [{pid}] Rotated {angle}° CW via {method} (conf: {score:.2f})", flush=True)
    return (pid, angle, method)


def auto_rotate_all_crops(output_dir=OUTPUT_DIR, manifest_path=MANIFEST_FILE):
    """
    Auto-detects and corrects orientation for all exported crop images in parallel.
    """
    import concurrent.futures
    if not os.path.exists(manifest_path):
        return 0
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    net = load_orientation_models()
    
    print(f"Auto-rotating photo crops in '{output_dir}'...", flush=True)
    
    items = []
    for scan_name, scan_info in manifest["scans"].items():
        for photo in scan_info["photos"]:
            pid = photo["id"]
            crop_path = os.path.join(output_dir, f"{pid}.jpg")
            items.append((pid, crop_path, net))
            
    total_rotated = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        results = list(executor.map(process_single_auto_rotate, items))
        
    rotated_map = {r[0]: (r[1], r[2]) for r in results if r is not None}
    
    for scan_name, scan_info in manifest["scans"].items():
        for photo in scan_info["photos"]:
            pid = photo["id"]
            if pid in rotated_map:
                angle, method = rotated_map[pid]
                if angle != 0:
                    photo["rotation"] = (photo.get("rotation", 0) + angle) % 360
                    total_rotated += 1
                photo["rotation_method"] = method
                
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Auto-rotation complete: {total_rotated} photos rotated.", flush=True)
    return total_rotated


def batch_rotate_photos(rotations_map, output_dir=OUTPUT_DIR, manifest_path=MANIFEST_FILE):
    """
    Batch-rotates multiple cropped photos by specified angle deltas (+90, -90, 180).
    """
    if not os.path.exists(manifest_path):
        return 0, "Manifest file not found"
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    total_rotated = 0
    for scan_name, scan_info in manifest["scans"].items():
        for photo in scan_info["photos"]:
            pid = photo["id"]
            if pid in rotations_map:
                angle_delta = int(rotations_map[pid]) % 360
                if angle_delta != 0:
                    crop_path = os.path.join(output_dir, f"{pid}.jpg")
                    if os.path.exists(crop_path):
                        img = cv2.imread(crop_path)
                        if img is not None:
                            rotated_img = rotate_cv2_image(img, angle_delta)
                            cv2.imwrite(crop_path, rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                            photo["rotation"] = (photo.get("rotation", 0) + angle_delta) % 360
                            photo["rotation_method"] = "manual"
                            total_rotated += 1
                            
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    return total_rotated, f"Successfully saved rotations for {total_rotated} photos"


def rotate_single_photo(photo_id, angle_delta=90, output_dir=OUTPUT_DIR, manifest_path=MANIFEST_FILE):
    """
    Rotates a single photo by angle_delta degrees (+90 CW or -90 CCW).
    """
    crop_path = os.path.join(output_dir, f"{photo_id}.jpg")
    if not os.path.exists(crop_path):
        return False, f"Photo file {photo_id}.jpg not found"
        
    img = cv2.imread(crop_path)
    if img is None:
        return False, "Failed to read image"
        
    rotated_img = rotate_cv2_image(img, angle_delta)
    cv2.imwrite(crop_path, rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
            
        for scan_name, scan_info in manifest["scans"].items():
            for photo in scan_info["photos"]:
                if photo["id"] == photo_id:
                    photo["rotation"] = (photo.get("rotation", 0) + angle_delta) % 360
                    photo["rotation_method"] = "manual"
                    break
                    
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
    return True, f"Rotated {photo_id} by {angle_delta}°"


class CropRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Server Handler for Preview Web UI and REST API.
    """
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if path in ["/", "/index.html", "/preview", "/gallery"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(PREVIEW_HTML_FILE, "rb") as f:
                self.wfile.write(f.read())
            return
            
        elif path == "/api/manifest":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if os.path.exists(MANIFEST_FILE):
                with open(MANIFEST_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(json.dumps({"scans": {}}).encode("utf-8"))
            return
            
        elif path == "/api/image":
            scan_name = query.get("name", [""])[0]
            scan_path = os.path.join(INPUT_DIR, scan_name)
            if os.path.exists(scan_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(scan_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Image not found")
                return
                
        elif path == "/api/cropped_image":
            photo_id = query.get("id", [""])[0]
            crop_path = os.path.join(OUTPUT_DIR, f"{photo_id}.jpg")
            if os.path.exists(crop_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(crop_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Cropped photo not found")
                return
                
        elif path == "/api/export_debug":
            try:
                generate_debug_images(INPUT_DIR, DEBUG_DIR, MANIFEST_FILE)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Exported debug overlays"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        else:
            return super().do_GET()
            
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len)
        
        if path == "/api/save_manifest":
            try:
                data = json.loads(post_body.decode("utf-8"))
                with open(MANIFEST_FILE, "w") as f:
                    json.dump(data, f, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Manifest saved"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
            
        elif path == "/api/recrop":
            try:
                if post_body:
                    data = json.loads(post_body.decode("utf-8"))
                    with open(MANIFEST_FILE, "w") as f:
                        json.dump(data, f, indent=2)
                run_crop_all(INPUT_DIR, OUTPUT_DIR, MANIFEST_FILE)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Recropped all photos"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
            
        elif path == "/api/export_debug":
            try:
                if post_body:
                    data = json.loads(post_body.decode("utf-8"))
                    with open(MANIFEST_FILE, "w") as f:
                        json.dump(data, f, indent=2)
                generate_debug_images(INPUT_DIR, DEBUG_DIR, MANIFEST_FILE)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Exported debug overlays"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
            
        elif path == "/api/auto_rotate_all":
            try:
                count = auto_rotate_all_crops(OUTPUT_DIR, MANIFEST_FILE)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "rotated_count": count, "message": f"Auto-rotated {count} photos"}).encode("utf-8"))
            except Exception as e:
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
            
        elif path == "/api/rotate_photo":
            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
                photo_id = data.get("photo_id", "")
                angle = int(data.get("angle", 90))
                success, msg = rotate_single_photo(photo_id, angle, OUTPUT_DIR, MANIFEST_FILE)
                if success:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "message": msg}).encode("utf-8"))
                else:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": msg}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        elif path == "/api/batch_rotate_photos":
            try:
                data = json.loads(post_body.decode("utf-8")) if post_body else {}
                rotations = data.get("rotations", {})
                count, msg = batch_rotate_photos(rotations, OUTPUT_DIR, MANIFEST_FILE)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "count": count, "message": msg}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return


def start_server(port=PORT):
    """
    Launches local HTTP preview server.
    """
    if not os.path.exists(MANIFEST_FILE):
        build_manifest()
        
    print(f"\n=======================================================")
    print(f" Starting Photo Cropper Preview Server")
    print(f" URL: http://localhost:{port}")
    print(f" Press Ctrl+C to stop")
    print(f"=======================================================\n")
    
    server_address = ("", port)
    handler = CropRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(description="Batch Photo Scan Segmenter & Cropper")
    parser.add_argument("--detect", action="store_true", help="Run detection and generate crops_manifest.json")
    parser.add_argument("--crop", action="store_true", help="Export cropped photos based on crops_manifest.json")
    parser.add_argument("--auto-rotate", action="store_true", help="Auto-detect orientation and rotate cropped photos")
    parser.add_argument("--debug", action="store_true", help="Generate debug overlay images in ScanOldPhotosDebug/")
    parser.add_argument("--server", action="store_true", help="Start preview web server")
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="Expand crops by N pixels margin (default: 2)")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: detect -> crop -> auto-rotate -> server")
    parser.add_argument("--port", type=int, default=PORT, help="Port for web server (default: 8000)")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1 or args.all:
        build_manifest()
        run_crop_all(margin=args.margin)
        auto_rotate_all_crops()
        generate_debug_images()
        start_server(args.port)
    elif args.detect:
        build_manifest()
    elif args.crop:
        run_crop_all(margin=args.margin)
    elif args.auto_rotate:
        auto_rotate_all_crops()
    elif args.debug:
        generate_debug_images()
    elif args.server:
        start_server(args.port)


if __name__ == "__main__":
    main()
