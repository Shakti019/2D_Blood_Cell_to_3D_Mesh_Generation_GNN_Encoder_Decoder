from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import torch
import os
import logging
from pathlib import Path
from preprocess import process_image
from model import BloodCellHybridGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("PORT", 8000))
HOST            = os.getenv("HOST", "0.0.0.0")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_UPLOAD_MB   = int(os.getenv("MAX_UPLOAD_MB", 10))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
WEIGHTS_PATH    = os.getenv("WEIGHTS_PATH", str(Path(__file__).parent / "bloodcell_hybrid_gnn.pth"))

CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG',
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CytoGraph",
    description="GNN-powered blood cell classification with 3D cubic mesh projection.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Static files (sample gallery images bundled in backend) ──────────────────
STATIC_PATH = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
logger.info(f"Static files served from {STATIC_PATH}")

# ── Model load ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Loading model on {device} from {WEIGHTS_PATH}")

if not Path(WEIGHTS_PATH).exists():
    raise RuntimeError(f"Model weights not found at {WEIGHTS_PATH}")

model = BloodCellHybridGNN(node_features_dim=6, hidden_channels=96, num_classes=8).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()
logger.info("Model loaded successfully.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@app.get("/app", response_class=HTMLResponse)
async def read_app(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a JPEG or PNG image.")

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        data, batch, mesh_data = process_image(image_bytes, line_step=60)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image. Ensure it contains a visible blood cell.")

    data  = data.to(device)
    batch = batch.to(device)

    with torch.no_grad():
        cls_percentages, _ = model(data.x, data.edge_index, data.edge_attr, batch)
        percentages = cls_percentages[0].cpu().numpy().tolist()

    predictions   = {CLASSES[i]: round(percentages[i] * 100, 2) for i in range(8)}
    predicted_class = CLASSES[percentages.index(max(percentages))]

    return {
        "predicted_class": predicted_class,
        "confidence":      max(percentages),
        "percentages":     predictions,
        "mesh_data":       mesh_data,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
