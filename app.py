"""JSON API for the exported Yelp sentiment model.

The application loads an ``artifacts`` directory when present. If it is not
present, it extracts the first usable outputs archive in the project root into
``.model_cache``.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

import joblib
from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_CACHE = ROOT / ".model_cache"
ARCHIVE_NAME = ("outputs.zip",)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive while rejecting paths that escape destination."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        source.extractall(destination)


def _find_artifacts(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    for metadata in root.rglob("model.json"):
        candidate = metadata.parent
        if (candidate / "logreg.joblib").is_file():
            return candidate
    return None


def _archive_candidates() -> list[Path]:
    candidates = []
    for name in ARCHIVE_NAME:
        candidates.extend(ROOT.glob(name))
    candidates.extend(ROOT.glob("outputs*.zip"))
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def resolve_artifacts() -> Path:
    existing = _find_artifacts(ARTIFACTS)
    if existing:
        return existing

    for archive in _archive_candidates():
        extracted = MODEL_CACHE / archive.stem
        candidate = _find_artifacts(extracted)
        if candidate:
            return candidate
        if extracted.exists():
            shutil.rmtree(extracted)
        _safe_extract(archive, extracted)
        candidate = _find_artifacts(extracted)
        if candidate:
            return candidate

    raise FileNotFoundError(
        "No model artifacts found. Add artifacts/ or place outputs.zip in the project root."
    )


ARTIFACT_DIR = resolve_artifacts()
with (ARTIFACT_DIR / "model.json").open(encoding="utf-8") as file:
    META = json.load(file)
MODEL_BUNDLE = joblib.load(ARTIFACT_DIR / "logreg.joblib")
VECTORIZER = MODEL_BUNDLE["vec"]
MODEL = MODEL_BUNDLE["model"]
LABELS = META.get("labels", ["negative", "neutral", "positive"])

app = Flask(__name__)


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def predict_one(text: str) -> dict:
    cleaned = _clean_text(text)
    probabilities = MODEL.predict_proba(VECTORIZER.transform([cleaned]))[0]
    probability_map = dict(
        zip(LABELS, [round(float(value), 4) for value in probabilities])
    )
    return {
        "label": LABELS[int(probabilities.argmax())],
        "probabilities": probability_map,
        "weights": {"logreg": 1.0},
    }


def _text_from_payload(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return None
    text = _clean_text(payload["text"])
    return text or None


@app.get("/")
def service_info():
    return jsonify(
        {
            "service": "Yelp sentiment API",
            "status": "ok",
            "endpoints": {
                "GET /health": "Readiness and loaded artifact location",
                "GET /model": "Model labels and training metadata",
                "GET /weights": "Backward-compatible alias for model metadata",
                "POST /predict": 'Predict one review: {"text": "..."}',
                "POST /predict/batch": 'Predict up to 100 reviews: {"texts": ["...", "..."]}',
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True, "artifacts": str(ARTIFACT_DIR)})


@app.get("/model")
@app.get("/weights")
def model_info():
    return jsonify(
        {
            "model_type": META.get("model_type", "TF-IDF + Logistic Regression"),
            "labels": LABELS,
            "weights": {"logreg": 1.0},
            "best_C": META.get("best_C"),
            "vectorizer": META.get("vectorizer", {}),
            "preprocessing": META.get("preprocessing", {}),
            "test_metrics": META.get("test_metrics", {}),
            "artifact_directory": str(ARTIFACT_DIR),
        }
    )


@app.post("/predict")
def predict():
    text = _text_from_payload(request.get_json(silent=True))
    if text is None:
        return jsonify({"error": 'JSON body must contain a non-empty string field "text"'}), 400
    return jsonify(predict_one(text))


@app.post("/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True)
    texts = payload.get("texts") if isinstance(payload, dict) else None
    if not isinstance(texts, list) or not texts or len(texts) > 100:
        return jsonify({"error": '"texts" must be a non-empty array with at most 100 items'}), 400
    if any(not isinstance(text, str) or not _clean_text(text) for text in texts):
        return jsonify({"error": 'every item in "texts" must be a non-empty string'}), 400
    return jsonify({"predictions": [predict_one(text) for text in texts]})


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
