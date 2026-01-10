"""Tests for Cell - multi-context value storage."""

import pytest
from hush.core.states.cell import Cell, DEFAULT_CONTEXT


# ============================================================
# Test 1: Basic Cell Operations
# ============================================================

class TestBasicCellOperations:
    """Test basic Cell set/get operations."""

    def test_set_and_get_with_context(self):
        """Test setting and getting values with specific contexts."""
        cell = Cell(default_value=0)

        cell["loop1"] = 10
        cell["loop2"] = 20

        assert cell["loop1"] == 10
        assert cell["loop2"] == 20

    def test_none_context_uses_default(self):
        """Test that None context maps to DEFAULT_CONTEXT."""
        cell = Cell(default_value=0)

        cell[None] = 30

        assert cell[None] == 30
        assert cell[DEFAULT_CONTEXT] == 30

    def test_default_value_returned_for_missing_context(self):
        """Test that default_value is returned for missing contexts."""
        cell = Cell(default_value=42)

        assert cell["nonexistent"] == 42


# ============================================================
# Test 2: Version Tracking
# ============================================================

class TestVersionTracking:
    """Test Cell version/context tracking."""

    def test_versions_track_insertion_order(self):
        """Test that versions track context insertion order."""
        cell = Cell()

        cell["first"] = 1
        cell["second"] = 2
        cell["third"] = 3

        assert cell.versions == ["first", "second", "third"]

    def test_updating_existing_context_does_not_add_version(self):
        """Test that updating existing context doesn't add duplicate version."""
        cell = Cell()

        cell["ctx1"] = 1
        cell["ctx2"] = 2
        cell["ctx1"] = 10  # Update existing

        assert cell.versions == ["ctx1", "ctx2"]
        assert cell["ctx1"] == 10


# ============================================================
# Test 3: Get Latest
# ============================================================

class TestGetLatest:
    """Test get_latest() method."""

    def test_get_latest_returns_most_recent(self):
        """Test get_latest returns the most recently added context value."""
        cell = Cell(default_value=0)

        cell["first"] = 1
        cell["second"] = 2
        cell["third"] = 3

        assert cell.get_latest() == 3

    def test_get_latest_empty_cell_returns_default(self):
        """Test get_latest returns default when cell is empty."""
        cell = Cell(default_value=42)

        assert cell.get_latest() == 42


# ============================================================
# Test 4: Pop Context
# ============================================================

class TestPopContext:
    """Test pop_context() method."""

    def test_pop_context_removes_and_returns_value(self):
        """Test that pop_context removes context and returns its value."""
        cell = Cell(default_value=0)

        cell["ctx1"] = 10
        cell["ctx2"] = 20

        value = cell.pop_context("ctx1")

        assert value == 10
        assert "ctx1" not in cell
        assert cell.versions == ["ctx2"]

    def test_pop_context_missing_returns_default(self):
        """Test pop_context returns default for missing context."""
        cell = Cell(default_value=42)

        value = cell.pop_context("nonexistent")

        assert value == 42


# ============================================================
# Test 5: Delete Item
# ============================================================

class TestDeleteItem:
    """Test __delitem__ method."""

    def test_delitem_removes_context(self):
        """Test del cell['context'] removes the context."""
        cell = Cell()

        cell["ctx1"] = 10
        cell["ctx2"] = 20

        del cell["ctx1"]

        assert "ctx1" not in cell
        assert cell.versions == ["ctx2"]


# ============================================================
# Test 6: Contains
# ============================================================

class TestContains:
    """Test __contains__ method."""

    def test_contains_returns_true_for_existing(self):
        """Test 'in' operator for existing contexts."""
        cell = Cell()
        cell["existing"] = 1

        assert "existing" in cell

    def test_contains_returns_false_for_missing(self):
        """Test 'in' operator for missing contexts."""
        cell = Cell()

        assert "missing" not in cell


# ============================================================
# Test 7: Repr
# ============================================================

class TestRepr:
    """Test __repr__ method."""

    def test_repr_shows_contexts_and_latest(self):
        """Test repr shows contexts dict and latest."""
        cell = Cell()
        cell["ctx1"] = 10
        cell["ctx2"] = 20

        repr_str = repr(cell)

        assert "Cell" in repr_str
        assert "ctx2" in repr_str  # Latest should be shown
