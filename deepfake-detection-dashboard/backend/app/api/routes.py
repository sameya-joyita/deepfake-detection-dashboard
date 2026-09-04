"""Health, model-card and video-analysis endpoints."""

from pathlib import Path
import tempfile
import uuid

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.errors import ApiError, VideoReadError


api = Blueprint("api", __name__, url_prefix="/api")


def _boolean_form_value(name, default):
    raw = request.form.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ApiError(
            f"{name} must be true or false.", code="invalid_option"
        )
    return value in {"true", "1", "yes"}


@api.get("/health")
def health():
    service = current_app.extensions.get("analysis_service")
    if service is None:
        return jsonify(
            {
                "ready": False,
                "error": current_app.extensions.get(
                    "model_load_error", "Model service is unavailable."
                ),
            }
        ), 503
    return jsonify(service.health())


@api.get("/model-card")
def model_card():
    service = current_app.extensions.get("analysis_service")
    if service is None:
        raise ApiError(
            "Model service is unavailable.",
            status_code=503,
            code="model_unavailable",
        )
    return jsonify(service.bundle.metadata)


@api.post("/analyze")
def analyze():
    service = current_app.extensions.get("analysis_service")
    if service is None:
        raise ApiError(
            "Model service is unavailable. Check /api/health.",
            status_code=503,
            code="model_unavailable",
        )

    uploaded = request.files.get("file")
    if uploaded is None:
        raise ApiError("Multipart field 'file' is required.", code="missing_file")

    safe_name = secure_filename(uploaded.filename or "")
    if not safe_name:
        raise ApiError("A filename is required.", code="missing_filename")

    extension = Path(safe_name).suffix.lower()
    if extension not in current_app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        raise ApiError(
            "Unsupported video extension. Allowed: mp4, avi, mov and mkv.",
            code="unsupported_file_type",
        )

    include_gradcam = _boolean_form_value(
        "include_gradcam", current_app.config["ENABLE_GRADCAM"]
    )
    analysis_id = uuid.uuid4().hex
    upload_root = current_app.config["UPLOAD_ROOT"]

    with tempfile.TemporaryDirectory(
        prefix=f"analysis_{analysis_id}_", dir=upload_root
    ) as temporary_directory:
        video_path = Path(temporary_directory) / f"upload{extension}"
        uploaded.save(video_path)
        if video_path.stat().st_size == 0:
            raise ApiError("The uploaded file is empty.", code="empty_file")

        try:
            result = service.analyse_video(
                video_path=video_path,
                analysis_id=analysis_id,
                filename=safe_name,
                include_gradcam=include_gradcam,
            )
        except VideoReadError as error:
            raise ApiError(
                str(error), status_code=422, code="unreadable_video"
            ) from error

    return jsonify(result), 200
