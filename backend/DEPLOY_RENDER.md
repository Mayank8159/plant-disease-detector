# Render Free Tier Deployment

## Included setup
- FastAPI app entrypoint: `main:app`
- Health endpoint: `/health`
- Model files expected in backend root:
  - `model.tflite`
  - `labels.txt`

## Deploy options

### Option A: Blueprint (recommended)
1. Push this repo to GitHub.
2. In Render, choose **New + > Blueprint**.
3. Select this repository.
4. Render will read `render.yaml` from repo root.

### Option B: Manual Web Service
If you create the service manually:
- Root Directory: `backend`
- Build Command: `pip install --upgrade pip ; pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Required environment variables
- `PYTHON_VERSION=3.11.9`
- `MODEL_PATH=model.tflite`
- `LABELS_PATH=labels.txt`
- `CONFIDENCE_THRESHOLD=0.65`
- `ALLOWED_ORIGINS=https://your-frontend-domain.onrender.com,http://localhost:3000`

## Notes for Free Tier reliability
- Keep `num_threads=1` in interpreter (already configured).
- Avoid adding full TensorFlow in requirements to stay memory-light.
- If your frontend domain changes, update `ALLOWED_ORIGINS` and redeploy.
