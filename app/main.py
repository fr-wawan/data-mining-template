from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.core.database import Base, engine
from app.core.templates import templates


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # Cookie-based session for simple learning app (server-side rendered).
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)

    # Create tables (simple for local learning)
    Base.metadata.create_all(bind=engine)

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Make templates accessible (import side-effect already ok)
    _ = templates

    return app


app = create_app()
