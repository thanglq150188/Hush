"""Pytest configuration for hush-observability tests."""

import os


def pytest_configure(config):
    """Configure pytest environment.

    Sets terminal width for Rich console to avoid log truncation in pytest.
    """
    # Set terminal width for Rich console output
    os.environ["COLUMNS"] = "200"