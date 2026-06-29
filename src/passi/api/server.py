"""Web API server stub — reserved for future use."""

from __future__ import annotations

from fastapi import FastAPI


def create_app(config=None) -> FastAPI:  # noqa: ANN001
    """Create the FastAPI application (not yet implemented)."""
    app = FastAPI(title="PassiAgent API", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    async def root():
        return {"message": "PassiAgent API — reserved for future use"}

    return app


def main() -> None:
    """Entry point for passi-server command (not yet implemented)."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000)
