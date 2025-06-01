# dlfeat/__init__.py

from .dlfeat_lib import (
    DLFeatExtractor, 
    list_available_models, 
    run_self_tests, 
    MODEL_CONFIGS,
    __version__  # Expose the version from dlfeat_lib.py
)

# You can also define __all__ if you want to be explicit about exports
__all__ = [
    "DLFeatExtractor", 
    "list_available_models", 
    "run_self_tests", 
    "MODEL_CONFIGS",
    "__version__"
]