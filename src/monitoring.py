"""
This module contains system resource monitoring utilities for the
tool-calling fine-tuning project.

The module supports monitoring:

- CPU utilization
- System memory usage
- GPU memory usage
- Execution time

This module is responsible only for collecting system metrics.
MLflow logging is handled separately in tracking.py.
"""

import time

import psutil
import torch


# ============================================================
# EXECUTION TIMER
# ============================================================

def start_timer():
    """
    Start an execution timer.

    Returns:
        float:
            Start timestamp.
    """

    return time.time()


def get_elapsed_time(start_time):
    """
    Calculate elapsed execution time.

    Args:
        start_time (float):
            Timestamp returned by start_timer().

    Returns:
        float:
            Elapsed time in seconds.
    """

    return time.time() - start_time


# ============================================================
# CPU MONITORING
# ============================================================

def get_cpu_usage():
    """
    Get current CPU utilization.

    Returns:
        float:
            CPU usage percentage.
    """

    return psutil.cpu_percent(
        interval=None
    )


# ============================================================
# MEMORY MONITORING
# ============================================================

def get_memory_usage():
    """
    Get current system memory usage.

    Returns:
        dict:
            System memory metrics.
    """

    memory = psutil.virtual_memory()

    return {

        "memory_usage_percent":
            memory.percent,

        "memory_used_gb":
            memory.used / (1024 ** 3),

        "memory_available_gb":
            memory.available / (1024 ** 3)
    }


# ============================================================
# GPU MONITORING
# ============================================================

def get_gpu_memory_usage():
    """
    Get GPU memory usage metrics.

    Returns:
        dict:
            GPU memory metrics.

        Returns an empty dictionary if CUDA is unavailable.
    """

    if not torch.cuda.is_available():

        return {}


    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    total = (

        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory

        / (1024 ** 3)
    )


    return {

        "gpu_memory_allocated_gb":
            allocated,

        "gpu_memory_reserved_gb":
            reserved,

        "gpu_memory_total_gb":
            total,

        "gpu_memory_allocated_percent":
            (allocated / total) * 100
    }


# ============================================================
# SYSTEM METRICS
# ============================================================

def get_system_metrics():
    """
    Collect CPU, RAM, and GPU metrics.

    Returns:
        dict:
            Combined system resource metrics.
    """

    metrics = {

        "cpu_usage_percent":
            get_cpu_usage()
    }

    metrics.update(
        get_memory_usage()
    )

    metrics.update(
        get_gpu_memory_usage()
    )

    return metrics