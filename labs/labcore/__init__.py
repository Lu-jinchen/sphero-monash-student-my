"""Lab 1 support library: dynamics model zoo, controllers, system ID, logging.

Kept as a package next to lab1.py so the lab scripts can `import labcore`
without touching the installed `sphero_env` package.
"""

from labcore.models import (  # noqa: F401
    MODEL_REGISTRY,
    ModelConfig,
    build_model,
    list_models,
)

__all__ = ["MODEL_REGISTRY", "ModelConfig", "build_model", "list_models"]
