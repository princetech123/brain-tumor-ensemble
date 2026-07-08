# REST API Documentation

This document describes the HTTP endpoints exposed by the Flask web application backend. The server runs by default on `http://localhost:5001`.

---

## Endpoints Summary

| Endpoint | HTTP Method | Description | Content-Type |
|---|---|---|---|
| `/` | `GET` | Serves the home page dropzone UI | `text/html` |
| `/upload` | `POST` | Accepts MRI scan, runs preprocessing, executes model inference and XAI overlay generation | `multipart/form-data` |
| `/analytics` | `GET` | Serves model metrics comparison table and performance curves | `text/html` |
| `/about` | `GET` | Serves documentation on architecture and clinical context | `text/html` |

---

## Detailed Endpoint Specifications

### 1. Home Dashboard
* **URL:** `/`
* **Method:** `GET`
* **Response:** HTML layout (`index.html`) presenting dropzone and description card.

---

### 2. Clinical Upload & Inference
* **URL:** `/upload`
* **Method:** `POST`
* **Request Format:** Form-data
* **Parameters:**
  * `file`: The binary MRI image file. Supported extensions: `.png`, `.jpg`, `.jpeg`.
* **Processing Chain:**
  1. Saves the upload in `src/web/static/uploads/`.
  2. Runs OpenCV preprocessor (Gaussian Blur, CLAHE, Skull Stripping).
  3. Executes ResNet50, EfficientNet-B3, ViT-B/16, and Ensemble Fusion models on the preprocessed tensor.
  4. Generates Grad-CAM gradients and ViT attention rollout heatmaps.
  5. Color-codes overlays and writes them in `src/web/static/temp/`.
* **Response:** HTML layout (`result.html`) populated with the following variables:
  * `orig_img`: Filename of the uploaded slice.
  * `prep_img`: Filename of the preprocessed scan.
  * `resnet_overlay`: Filename of the ResNet50 Grad-CAM image.
  * `effnet_overlay`: Filename of the EfficientNet Grad-CAM image.
  * `vit_overlay`: Filename of the ViT attention map image.
  * `res_prob`: ResNet50 confidence percentage.
  * `eff_prob`: EfficientNet-B3 confidence percentage.
  * `vit_prob`: ViT confidence percentage.
  * `fusion_prob`: Attention Ensemble confidence percentage.
  * `resnet_weight`: Dynamic fusion weight allocated to ResNet50 (%).
  * `effnet_weight`: Dynamic fusion weight allocated to EfficientNet (%).
  * `vit_weight`: Dynamic fusion weight allocated to ViT (%).
  * `latency`: System execution time (ms).
  * `classification`: Diagnosis string ("Tumor Detected" or "Normal").

---

### 3. Performance Analytics
* **URL:** `/analytics`
* **Method:** `GET`
* **Flow:** 
  1. Attempts to read classification reports from `logs/metrics.json`.
  2. Falls back to simulated metrics if training logs are missing.
* **Response:** HTML page (`analytics.html`) rendering ROC curves, confusion matrices, and the comparison table.

---

### 4. About Panel
* **URL:** `/about`
* **Method:** `GET`
* **Response:** HTML page (`about.html`) presenting project documentation.
