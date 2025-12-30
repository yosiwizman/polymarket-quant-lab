"""FastAPI application for local operator console."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pmq import __version__
from pmq.web.routes import router

# Template and static directories
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


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

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Favicon route
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        """Serve favicon."""
        return FileResponse(
            STATIC_DIR / "favicon.svg",
            media_type="image/svg+xml",
        )

    # Chrome DevTools well-known route (silences 404 noise)
    @app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
    async def devtools_well_known() -> Response:
        """Return empty response for Chrome DevTools."""
        return Response(status_code=204)

    # Include routes
    app.include_router(router)

    return app
