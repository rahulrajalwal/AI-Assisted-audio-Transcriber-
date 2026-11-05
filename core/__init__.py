"""Core backend package for the AI Meeting Assistant.

Importing anything from ``core`` loads ``.env`` first (see ``core.config``).
"""

from core.config import load_environment

load_environment()
