"""FastAPI application with lifespan management."""

import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.openapi.utils import get_openapi  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from agent_runtime.agents.runtime import get_db  # noqa: E402
from agent_runtime.api.auth import require_bearer_token  # noqa: E402
from agent_runtime.api.routers import models, prompts, sessions  # noqa: E402
from agent_runtime.db.prompt_repo import SystemPromptRepo  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect DB, seed default prompt
    db = await get_db()
    prompt_repo = SystemPromptRepo(db)
    await prompt_repo.seed_default()
    yield
    # Shutdown: disconnect DB
    await db.disconnect()


app = FastAPI(
    title="Agent Runtime",
    description="Agent runtime API powered by OpenAI Agents SDK",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, dependencies=[Depends(require_bearer_token)])
app.include_router(prompts.router, dependencies=[Depends(require_bearer_token)])
app.include_router(models.router, dependencies=[Depends(require_bearer_token)])


@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/openapi.json", include_in_schema=False)
async def openapi_schema(_: str = Depends(require_bearer_token)):
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui(token: str = Depends(require_bearer_token)):
    token_json = json.dumps(token)
    title_json = json.dumps(f"{app.title} - Swagger UI")
    return HTMLResponse(
        f"""
<!DOCTYPE html>
<html>
<head>
  <link type="text/css" rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <title>{app.title} - Swagger UI</title>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({{
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      layout: "BaseLayout",
      deepLinking: true,
      showExtensions: true,
      showCommonExtensions: true,
      requestInterceptor: function(request) {{
        request.headers["Authorization"] = "Bearer " + {token_json};
        return request;
      }},
      documentTitle: {title_json}
    }});
  </script>
</body>
</html>
"""
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_ui(_: str = Depends(require_bearer_token)):
    schema_json = json.dumps(
        get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
    )
    return HTMLResponse(
        f"""
<!DOCTYPE html>
<html>
<head>
  <title>{app.title} - ReDoc</title>
</head>
<body>
  <redoc></redoc>
  <script src="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js"></script>
  <script>
    Redoc.init({schema_json}, {{}}, document.querySelector("redoc"));
  </script>
</body>
</html>
"""
    )
