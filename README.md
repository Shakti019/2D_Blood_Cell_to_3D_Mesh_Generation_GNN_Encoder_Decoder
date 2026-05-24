# CytoGraph: 2D Blood Cell to 3D Mesh Generation & GNN Encoder-Decoder

**CytoGraph** is a cutting-edge computer vision and graph-based machine learning pipeline that classifies microscopic blood cell images into 8 distinct cell types. Moving beyond the limitations of traditional 2D Convolutional Neural Networks (CNNs), CytoGraph pioneers a novel morpho-spatial approach. It dynamically translates the 2D footprint of a blood cell's nucleus into a **3D Cubic Mesh Graph** and classifies it using a custom **Multi-Task Graph Attention Network (GATv2)**.

This repository contains both the exploratory Jupyter Notebooks used for mathematical modeling and architecture design, and a fully deployable FastAPI web application for real-time inference.
<img width="700" height="551" alt="nucleus_3d3" src="https://github.com/user-attachments/assets/279e014c-9690-4da8-913e-2e7c3ab16e7c" />


---

## 🔬 Supported Blood Cell Types (8 Classes)
The model is trained to recognize the following classes from standard microscopy imagery:
1. **Basophil** – Rare WBC with large dark-staining granules.
2. **Eosinophil** – Bilobed nucleus, red-orange granules.
3. **Erythroblast** – Immature red blood cell precursor.
4. **IG (Immature Granulocyte)** – Band-shaped or horseshoe nucleus.
5. **Lymphocyte** – Small WBC, large round dark nucleus.
6. **Monocyte** – Largest WBC, kidney-shaped nucleus.
7. **Neutrophil** – Most common WBC, multi-lobed nucleus.
8. **Platelet** – Tiny cell fragment, lacking a nucleus.

---

## 🚀 Theoretical Approach & Pipeline Architecture

The core philosophy of CytoGraph is that the structural topology, precise shape contour, and chromatin density of a cell's nucleus can be better represented as an interconnected spatial graph rather than a flat matrix of pixels. 

### Phase 1: 2D Data Preprocessing & Nucleus Isolation
Instead of feeding full images directly into a neural network, the application extracts the biological structures that matter most:
* **Color Space Transformation:** The raw BGR image is converted into HSV color space. We specifically target the Saturation (S) channel, which highly reacts to deeply stained genetic material (chromatin).
* **Otsu Thresholding & Morphology:** A Gaussian blur paired with automatic Otsu thresholding isolates the nucleus. Morphological opening and closing operations with a 5x5 kernel clean the binary mask, ensuring the extracted boundary is free of scattered artifacts.
* **Pixel Extraction:** The bounding box and all intra-nuclear RGB arrays and their true 2D coordinates are extracted for the next phase.

### Phase 2: 3D Cubic Mesh Generation
Once the 2D nucleus footprint is extracted, we project it into a spatial 3D paradigm:
* **6-Face Extrapolation:** The 2D coordinates are symmetrically projected onto a virtual 3D cube configuration (Front, Back, Right, Left, Top, Bottom faces) centered around the cell. 
* **Node Generation:** The 2D pixels are subsampled spatially. Each node represents a 3D point containing spatial and color data: `[X, Y, Z, R_norm, G_norm, B_norm]`. The coordinates are Z-score normalized.
* **Delaunay Triangulation:** We generate a continuous surface across the nodes using Delaunay triangulation. This maps 2D proximity into 3D structural integrity.

### Phase 3: Spatial Graph Embedding (Edges)
To define the topological relationships for the Graph Neural Network:
* Edges are routed between spatial nodes.
* Each edge is embedded with 4 key attributes:
  1. `Euclidean Distance` between Node $A$ and Node $B$.
  2. `Average Red (R)` across the edge.
  3. `Average Green (G)` across the edge.
  4. `Average Blue (B)` across the edge.

### Phase 4: Multi-Task Graph Neural Network
The core intelligence engine uses a custom architecture built on **PyTorch Geometric**.

* **The Encoder (GATv2Conv):**
  We utilize a 3-block Multi-Head **GATv2Conv** (Graph Attention Network v2). Unlike standard GATs, traditional static attention limits the modeling capacity of nodes. GATv2 introduces dynamic attention, scaling dynamically and preventing attention dilution across the rich 3D mesh.
  * *Block 1 & 2:* 4-head attention, stabilized dynamically via `GraphNorm` and activated with `GELU` (Gaussian Error Linear Units) for deeper gradient propagation.
  * *Block 3:* Single-head attention reducing down to a latent node feature representation.

* **Dual-Head Output:**
  * **Head 1 (Classification Decoder):** Extracts a global cell spatial signature by concatenating `global_mean_pool` and `global_max_pool` representations. A deep MLP (Linear -> BatchNorm -> GELU -> Dropout) calculates the logit probabilities for the 8 target cell types.
  * **Head 2 (Anomaly & Structure Decoder):** A secondary linear pathway attempts to reconstruct the original node matrices. This acts as an auto-encoder style regularizer, heavily enriching the learned latent representations in the primary encoder and allowing it to intimately "understand" 3D cellular structure.

---

## 🛠️ Tech Stack & Ecosystem

- **Machine Learning & Graphs:** PyTorch 2.0+, PyTorch Geometric (PyG 2.7)
- **Computer Vision:** OpenCV (`cv2`), SciPy, Matplotlib (Triangulation & 3D Projections)
- **Backend API:** FastAPI (Async workflows), Uvicorn
- **Frontend & Rendering:** HTML5, Jinja2 Templates, Plotly.js (WebGL 3D surface rendering), TailwindCSS
- **Model format:** State dictionary loaded onto CPU/CUDA dynamically.

---

## 📁 Repository Structure

```text
CytoGraph/
├── backend/                       # Production web server & API
│   ├── main.py                    # FastAPI server & route handlers
│   ├── model.py                   # PyTorch GATv2 custom model architecture
│   ├── preprocess.py              # Vision pipeline: OpenCV → Graph 3D Mesh
│   ├── bloodcell_hybrid_gnn.pth   # Pre-trained Model Weights
│   ├── requirements.txt           # Python dependencies (FastAPI, PyTorch, PyG)
│   ├── Procfile                   # Heroku / Render deployment config
│   ├── .env.example               # Environment variables template
│   ├── static/                    # Sample blood cell datasets bundled for UI
│   └── templates/                 # UI HTML / Jinja logic
├── bloodcells_dataset/            # Original RGB training images (Categorized)
├── DataPreprocessing.ipynb        # Mathematical development of the 3D projection
├── Model.ipynb                    # GNN dataset generation and model training loops
└── training_history.csv           # Model metrics telemetry across epochs
