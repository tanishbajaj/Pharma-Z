"""
Pharma-Z — FastAPI Application Entry Point

Registers all routers, configures CORS, and creates database tables.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, Base
from app.api.routes.health_routes import health_router


# ---------------------------------------------------------------------------
# Lifespan — runs on startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Create database tables on startup."""
    Base.metadata.create_all(bind=engine)
    yield


# ---------------------------------------------------------------------------
# Application Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-Preserving, Blockchain-Enabled AI Drug Discovery & Forensic Audit Platform",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS — allow all origins in development
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Register Routers
# ---------------------------------------------------------------------------

API_V1_PREFIX = "/api/v1"

app.include_router(health_router, prefix=API_V1_PREFIX, tags=["Health"])

# --- Layer 1: AI Pipeline ---
from app.api.routes.molecule_routes import molecule_router
from app.api.routes.binding_routes import binding_router
from app.api.routes.toxicity_routes import toxicity_router
from app.api.routes.pipeline_routes import pipeline_router

app.include_router(molecule_router, prefix=API_V1_PREFIX, tags=["Molecules"])
app.include_router(binding_router, prefix=API_V1_PREFIX, tags=["Binding Affinity"])
app.include_router(toxicity_router, prefix=API_V1_PREFIX, tags=["Toxicity"])
app.include_router(pipeline_router, prefix=API_V1_PREFIX, tags=["Pipeline"])

# --- Layer 1b: GraphVAE (Stage 2 Molecule Generation) ---
from app.api.routes.graph_molecule_routes import graph_molecule_router
app.include_router(graph_molecule_router, prefix=API_V1_PREFIX, tags=["Graph Molecules"])

# --- Layer 2: Audit ---
from app.api.routes.audit_routes import audit_router
app.include_router(audit_router, prefix=API_V1_PREFIX, tags=["Audit"])

# --- Layer 4: Blockchain ---
from app.api.routes.blockchain_routes import blockchain_router
app.include_router(blockchain_router, prefix=API_V1_PREFIX, tags=["Blockchain"])
