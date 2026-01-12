"""
CUDA GPU-accelerated market simulation.

This package provides a fully GPU-accelerated implementation of the market simulator
using CuPy/CUDA for maximum throughput.

Target performance:
- >25,000 steps/s (100 msgs, 1 agent) on RTX 4090
- >400,000 steps/s (1 msg, 1 agent) on RTX 4090
- Scales to multiple H100s
"""

import warnings

# Check for CuPy availability
_CUPY_AVAILABLE = False
_CUDA_VERSION = None
_GPU_NAME = None
_GPU_COUNT = 0

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    _CUDA_VERSION = cp.cuda.runtime.runtimeGetVersion()
    _GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    if _GPU_COUNT > 0:
        with cp.cuda.Device(0):
            props = cp.cuda.runtime.getDeviceProperties(0)
            _GPU_NAME = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
except ImportError:
    warnings.warn(
        "CuPy not available. Install with: pip install cupy-cuda12x\n"
        "GPU acceleration will not be available."
    )
except Exception as e:
    warnings.warn(f"CUDA initialization failed: {e}")


def is_available() -> bool:
    """Check if CUDA GPU acceleration is available."""
    return _CUPY_AVAILABLE and _GPU_COUNT > 0


def get_device_count() -> int:
    """Get number of available CUDA devices."""
    return _GPU_COUNT


def get_cuda_version() -> int:
    """Get CUDA runtime version."""
    return _CUDA_VERSION


def get_gpu_name() -> str:
    """Get name of the primary GPU."""
    return _GPU_NAME


def get_device_info() -> dict:
    """Get detailed information about available GPUs."""
    if not _CUPY_AVAILABLE:
        return {"available": False, "error": "CuPy not installed"}

    if _GPU_COUNT == 0:
        return {"available": False, "error": "No CUDA devices found"}

    import cupy as cp

    devices = []
    for i in range(_GPU_COUNT):
        with cp.cuda.Device(i):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            devices.append({
                "id": i,
                "name": name,
                "total_memory_gb": props['totalGlobalMem'] / (1024**3),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "multiprocessors": props['multiProcessorCount'],
            })

    return {
        "available": True,
        "cuda_version": _CUDA_VERSION,
        "device_count": _GPU_COUNT,
        "devices": devices,
    }


def print_device_info():
    """Print information about available GPUs."""
    info = get_device_info()

    if not info["available"]:
        print(f"CUDA not available: {info.get('error', 'Unknown error')}")
        return

    print(f"CUDA Version: {info['cuda_version']}")
    print(f"Number of GPUs: {info['device_count']}")
    print()

    for dev in info["devices"]:
        print(f"GPU {dev['id']}: {dev['name']}")
        print(f"  Memory: {dev['total_memory_gb']:.1f} GB")
        print(f"  Compute Capability: {dev['compute_capability']}")
        print(f"  Multiprocessors: {dev['multiprocessors']}")
        print()


# Public API
__all__ = [
    'is_available',
    'get_device_count',
    'get_cuda_version',
    'get_gpu_name',
    'get_device_info',
    'print_device_info',
]

# Lazy imports for main classes
def __getattr__(name):
    if name == 'CUDASimulator':
        from .simulator import CUDASimulator
        return CUDASimulator
    elif name == 'GPUOrderBook':
        from .order_book import GPUOrderBook
        return GPUOrderBook
    elif name == 'MultiGPUSimulator':
        from .multi_gpu import MultiGPUSimulator
        return MultiGPUSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
