"""
main.py — FastAPI application entry point.
===========================================
Assembles routers, configures CORS, adds exception handlers,
and sets up the OpenAPI docs.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router_network import router as network_router
from backend.api.router_agent import router as agent_router
from backend.api.router_routing import router as routing_router
from backend.api.router_metrics import router as metrics_router
from backend.services.routing_service import get_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initialise the routing service on startup."""
    logger.info("🚀 Starting AI Packet Routing API")
    svc = get_service()  # creates the singleton + default network
    logger.info(
        "Network ready: %d nodes, %d edges",
        svc.network.graph.number_of_nodes(),
        svc.network.graph.number_of_edges(),
    )
    yield
    logger.info("👋 Shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-Driven Adaptive Packet Routing API",
    description=(
        "REST API for a Q-Learning–based adaptive routing system. "
        "Simulates a packet-switched network, trains an RL agent, "
        "and compares performance against static Dijkstra routing."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc
)


# ---------------------------------------------------------------------------
# CORS — allow React frontend (dev: localhost:5173, prod: any)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # Vite dev server
        "http://localhost:3000",       # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=400,
        content={"error": "Operation Error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

app.include_router(network_router)
app.include_router(agent_router)
app.include_router(routing_router)
app.include_router(metrics_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    """Root health check endpoint."""
    svc = get_service()
    return {
        "status": "healthy",
        "service": "AI Packet Routing API",
        "version": "1.0.0",
        "network": {
            "nodes": svc.network.graph.number_of_nodes(),
            "edges": svc.network.graph.number_of_edges(),
        },
        "agent_trained": svc.is_trained,
    }
