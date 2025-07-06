
# 📝 Player Re-Identification: Cross-Camera Mapping — Report

## 🧠 Approach & Methodology

This project aims to maintain consistent player identities across different camera views (broadcast and tacticam) in sports videos. The solution is built as a modular pipeline with the following stages:

1. **Player Detection:**  
   - Used a fine-tuned **YOLOv8** model (`best.pt`) for detecting players and ball in both video streams.

2. **Multi-Object Tracking:**  
   - **ByteTrack** is applied to create unique tracking IDs across frames in each video.

3. **Feature Extraction:**  
   - Extracted player embeddings using **OSNet** (`osnet_x1_0`) from `torchreid`.
   - Bounding boxes of players were cropped, resized to 128x256, and used for appearance-based feature encoding.

4. **Cross-View Mapping:**  
   - Constructed a **cosine similarity matrix** between player embeddings from both cameras.
   - **Hungarian algorithm** computed the optimal assignment with a similarity threshold of 0.5.

5. **Visualization & Export:**  
   - Rendered bounding boxes and IDs on output videos.
   - Saved results in `player_mapping.json`, `broadcast_result.mp4`, and `tacticam_result.mp4`.

---

## ⚙️ Techniques Tried & Outcomes

| Technique                           | Outcome                                              |
|----------                           |---------                                             |
| YOLOv8 + ByteTrack                  | Reliable tracking, some drift during occlusion       |
| OSNet feature aggregation           | Improved viewpoint robustness, consistent embeddings |
| Cosine Similarity + Hungarian       | Effective cross-view ID assignment                   |
| Frame skipping during visualization | Reduced rendering time with minimal loss of clarity  |

---

## 🚧 Challenges Encountered

- **Viewpoint Variability:**  
  Player appearance changes drastically across views. Resolved by averaging features across frames and using OSNet for viewpoint-invariant embedding.

- **Occlusion & Partial Visibility:**  
  Caused frequent ID switches. Lowering YOLOv8 confidence threshold and using ByteTrack helped mitigate this.

- **Player Similarity:**  
  In sports, many players wear similar/identical jerseys. Without jersey number recognition, visual distinction remains a challenge.

- **Latency & Runtime:**  
  Feature extraction and similarity matching take time. Optimizations like frame skipping and batch processing improved runtime.

---

## ⚡ Performance & Hardware Notes

- The entire pipeline took approximately **12 minutes** to run on a **CPU-only setup**.
- With **GPU acceleration** (using **CUDA**, **cuDNN**, **PyTorch**, or **TensorFlow**), this can be executed significantly faster—enabling near real-time processing or handling longer videos with less delay.

---

## ✅ Summary

This solution offers a reproducible and modular framework for cross-camera player re-identification. It balances detection accuracy, appearance-based matching, and visual output in a relatively simple pipeline. With further work, it can be expanded for real-time, production-level deployment.

---

**Author:** Eshwar Adduri
**Date:** 25-06-2025
