"""
test_app_smoke.py - Smoke Tests for Streamlit Frontend

Uses `streamlit.testing.v1.AppTest` to verify that the app loads,
all core widgets exist (text area, length selectbox, temperature slider, buttons),
and no uncaught exceptions are raised on startup.
"""

from __future__ import annotations

import unittest
from streamlit.testing.v1 import AppTest


class TestAppSmoke(unittest.TestCase):
    """Smoke test suite using Streamlit AppTest framework."""

    def test_app_loads_without_exceptions(self) -> None:
        """App should initialize without raising unhandled exceptions."""
        at = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(at.exception, f"App raised unexpected exception: {at.exception}")

    def test_core_widgets_present(self) -> None:
        """Verifies text area, buttons, selectbox, and sliders are rendered."""
        at = AppTest.from_file("app.py", default_timeout=30).run()

        # Check title
        self.assertTrue(len(at.title) >= 1)
        self.assertIn("Story Continuation", at.title[0].value)

        # Check prompt text_area
        self.assertTrue(len(at.text_area) >= 1)

        # Check buttons (Generate and Reset)
        button_labels = [b.label for b in at.button]
        self.assertTrue(any("Generate" in lbl for lbl in button_labels))
        self.assertTrue(any("Reset" in lbl for lbl in button_labels))

        # Check selectbox (Length Presets)
        self.assertTrue(len(at.selectbox) >= 1)

        # Check slider (Temperature)
        self.assertTrue(len(at.slider) >= 1)


if __name__ == "__main__":
    unittest.main()
