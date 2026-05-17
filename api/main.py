from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.database import create_db
from api.routers import days, load_score, profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    yield


app = FastAPI(title="AI Coach API", lifespan=lifespan)

app.include_router(days.router,       prefix="/api/days",       tags=["days"])
app.include_router(load_score.router, prefix="/api/load-score", tags=["load-score"])
app.include_router(profile.router,    prefix="/api/profile",    tags=["profile"])

_HTML = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML.read_text(encoding="utf-8")
