"""FastAPI application for local operator console."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from pmq import __version__
from pmq.web.routes import router

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(
        title="Polymarket Quant Lab",
        description="Local operator console for paper trading (read-only)",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
    )

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app.state.templates = templates

    # Include routes
    app.include_router(router)

    return app
