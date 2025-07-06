
# 🏃‍♂️ Player Re-Identification: Cross-Camera Player Mapping

## 📌 Overview

This project implements a solution for **player re-identification across multiple camera views** in sports footage. Given two videos from different camera angles (`broadcast.mp4` and `tacticam.mp4`), the system ensures that each player retains a **consistent ID** across both views using computer vision techniques and deep learning.

---

## 🚀 Key Features

- 🎯 **Fine-tuned YOLOv8** model for player/ball detection  
- 📹 **ByteTrack** for robust player tracking across frames  
- 🔍 **OSNet** for appearance-based feature extraction  
- ↔️ **Cross-camera mapping** using cosine similarity + Hungarian algorithm  
- 📊 **Visualized results** with consistent player IDs  

---

## 📁 Project Structure

```
├── videos/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── main.py
├── player_mapping.json
├── broadcast_result.mp4
├── tacticam_result.mp4
├── best.pt
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Download Model Weights

Place the YOLOv8 model file (`best.pt`) in the project root directory.

> 📥 **Download Link:** [Download best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

---

## 🧠 How to Run

Run the main pipeline with:

```use the notebook and run the cell
main.ipynb
```

### Output Files

- `player_mapping.json` – Mapping between player IDs across cameras  
- `broadcast_result.mp4` – Broadcast view with consistent player IDs  
- `tacticam_result.mp4` – Tacticam view with consistent player IDs  

---

## 🧪 Methodology

### 📷 Player Detection & Tracking

- Model: YOLOv8 + ByteTrack  
- Confidence Threshold: `0.3` (to detect occluded players)  
- Tracks players persistently across frames  

### 🧬 Feature Extraction

- Network: OSNet (`osnet_x1_0`) from `torchreid`  
- Input: 128x256 RGB crops resized from bounding boxes  
- Output: Aggregated feature vector per track  

### 🔁 Cross-Camera Matching

- Metric: Cosine similarity  
- Assignment: Hungarian algorithm  
- Threshold: Similarity > `0.5` for valid matches  

### 🎨 Visualization

- ✅ Green box: Successfully mapped player  
- ❌ Red box: Unmapped player  
- 🆔 Displayed player ID on bounding box  

---

## 🧩 Challenges & Solutions

### Occlusion Handling

- Used ByteTrack for persistent ID tracking  
- Lowered detection threshold to detect partially visible players  

### Viewpoint Variation

- Aggregated features across time  
- Used viewpoint-invariant OSNet model  

### Lighting Differences

- Feature normalization before similarity comparison  
- Used cosine similarity instead of Euclidean distance  

### Performance Optimization

- Batched feature extraction  
- Frame skipping during visualization  

---
## Performance & Hardware Notes

- The entire pipeline took approximately **12 minutes** to run on a **CPU-only setup**.
- If executed with **GPU acceleration** (via CUDA, cuDNN, PyTorch, or TensorFlow), runtime can be significantly reduced, allowing near real-time inference and handling of longer videos more efficiently.
---

## 📦 Dependencies

Tested with **Python 3.8+**

Install with:

```bash
pip install -r requirements.txt
```

---
