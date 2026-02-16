"""
api — FastAPI router package.
"""

from .router_network import router as network_router
from .router_agent import router as agent_router
from .router_routing import router as routing_router
from .router_metrics import router as metrics_router
