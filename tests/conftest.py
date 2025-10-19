"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment once per session."""
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    yield

    # Cleanup if needed
    pass
