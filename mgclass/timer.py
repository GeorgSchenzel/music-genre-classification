import time
from dataclasses import dataclass


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    """
    A simple utility to time a section of code.

    Usage:
    with Timer("Task"):
        ...
    """

    label: str

    def __init__(self, label: str = ""):
        self._start_time = None
        self.label = label

    def start(self):
        """Start a new timer"""

        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        return self

    def stop(self, prnt=True):
        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if prnt:
            print(f"{self.label} finished in: {elapsed_time:0.4f} seconds")
        return elapsed_time

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()
