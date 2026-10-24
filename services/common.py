"""Shared service helpers: JSON error handling, request ids, and a production WSGI server (waitress)."""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from shelfai.errors import ShelfAIError

log = logging.getLogger("shelfai")


def setup_logging():
    logging.basicConfig(level=os.environ.get("SHELFAI_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


def register_error_handlers(app: Flask):
    app.json.sort_keys = False  # keep the documented, logical key order (status and summary first)
    @app.errorhandler(ShelfAIError)
    def _shelfai(e: ShelfAIError):
        return jsonify(e.to_dict(request.headers.get("X-Request-ID"))), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(e):
        mb = (app.config.get("MAX_CONTENT_LENGTH") or 0) // (1024 * 1024)
        return jsonify({"status": "error", "error": {"code": "IMAGE_TOO_LARGE",
                                                     "message": f"The upload is larger than {mb} MB."}}), 413

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        return jsonify({"status": "error", "error": {"code": e.name.upper().replace(" ", "_"),
                                                     "message": e.description}}), e.code

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        log.exception("Unhandled error")
        return jsonify({"status": "error", "error": {"code": "INTERNAL_ERROR", "message": str(e)}}), 500


def serve(app: Flask, port: int, threads: int = 8):
    """waitress: a production WSGI server that also works on Windows (gunicorn does not)."""
    from waitress import serve as waitress_serve
    host = os.environ.get("SHELFAI_HOST", "0.0.0.0")
    log.info("serving %s on http://%s:%d", app.name, host, port)
    waitress_serve(app, host=host, port=port, threads=threads, max_request_body_size=64 * 1024 * 1024)
