#!/usr/bin/env python
"""Local server for the recommendations widget.

    python scripts/serve.py                 # http://localhost:8000
    python scripts/serve.py --port 8080
    uvicorn scripts.serve:app --reload       # for development

Serves:
  - the existing web/ directory as static files (bioage_console.html,
    console.template.html, recommend.html, and their assets), so the
    calculator and the recommendations widget share one local origin and
    the "Get recommendations" link needs no CORS workaround
  - POST /api/analyze -- the BioAgeHack-specific endpoint: raw markers in,
    score + recommendations out (bioage.recommendations.analyze_router)
  - POST /api/recommendations[/preview] -- the generic, provider-agnostic
    endpoint that takes an already-built HealthProfile
    (bioage.recommendations.router)

Opening web/bioage_console.html directly from disk still works exactly as
before and needs no server -- this only adds the recommendations path, which
is honest about needing one. See bioage/recommendations/README or the parent
repo's docs for what a recommendations call costs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from bioage import config as C  # noqa: E402
from bioage.recommendations.analyze_router import router as analyze_router  # noqa: E402
from bioage.recommendations.router import router as recommendations_router  # noqa: E402

app = FastAPI(title="BioAgeHack recommendations")

# Local demo tool, not a deployed service -- permissive CORS is deliberate so
# the widget works whether the console was opened from disk (Origin: null) or
# served from this same process. Tighten this before putting the server
# anywhere reachable off the demo machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(recommendations_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# Mounted last: FastAPI matches routes in registration order, and a static
# mount at "/" would otherwise shadow the API routes above.
app.mount("/", StaticFiles(directory=str(C.ROOT / "web"), html=True), name="web")


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    print(f"Console:  http://{args.host}:{args.port}/bioage_console.html")
    print(f"API:      http://{args.host}:{args.port}/api/analyze")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
