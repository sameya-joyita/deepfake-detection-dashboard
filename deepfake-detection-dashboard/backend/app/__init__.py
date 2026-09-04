"""Flask application factory.

Imports are intentionally local so pure evidence/narrative modules can be used
by offline tools without importing the web framework.
"""


def create_app(config_override=None, analysis_service=None):
    import logging
    from pathlib import Path

    from flask import Flask, jsonify
    from flask_cors import CORS
    from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

    from app.api.routes import api
    from app.config import Config, validate_config
    from app.errors import ApiError

    app = Flask(__name__)
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)
    validate_config(app.config)

    Path(app.config["UPLOAD_ROOT"]).mkdir(parents=True, exist_ok=True)
    CORS(
        app,
        resources={r"/api/*": {"origins": app.config["FRONTEND_ORIGINS"]}},
    )

    app.extensions["analysis_service"] = analysis_service
    app.extensions["model_load_error"] = None

    if analysis_service is None and app.config.get("LOAD_MODELS", True):
        try:
            from app.services.analysis import AnalysisService

            app.extensions["analysis_service"] = AnalysisService.from_config(
                app.config
            )
        except Exception as error:
            app.logger.exception("Model service failed to initialise.")
            app.extensions["model_load_error"] = (
                "Required model artifacts could not be loaded."
            )
            if not app.config.get("TESTING", False):
                logging.getLogger(__name__).warning(
                    "The API started in not-ready state: %s", error
                )

    app.register_blueprint(api)

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify(
            {"error": {"code": error.code, "message": error.message}}
        ), error.status_code

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(error):
        return jsonify(
            {
                "error": {
                    "code": "file_too_large",
                    "message": "The uploaded video exceeds the configured limit.",
                }
            }
        ), 413

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        return jsonify(
            {
                "error": {
                    "code": error.name.lower().replace(" ", "_"),
                    "message": error.description,
                }
            }
        ), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception("Unhandled API error.")
        return jsonify(
            {
                "error": {
                    "code": "internal_error",
                    "message": "The analysis could not be completed.",
                }
            }
        ), 500

    return app
