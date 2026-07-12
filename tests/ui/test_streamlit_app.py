import pytest
from streamlit.testing.v1 import AppTest

def test_streamlit_ui_loads():
    """
    Test that the Streamlit application starts successfully without any
    uncaught exceptions and renders the main components.
    """
    at = AppTest.from_file("app/ui/streamlit_app.py")
    # Run the app simulating a typical run
    at.run()
    
    # Ensure there are no runtime exceptions during rendering
    assert not at.exception, f"Streamlit app raised an exception: {at.exception}"
    
    # Check that the main title exists
    assert "⚙️ PyRAG Developer Console" in [title.value for title in at.title]
    
    # Check that the tabs are rendered correctly
    tabs = at.tabs
    assert len(tabs) >= 6, f"Expected at least 6 tabs, found {len(tabs)}"
    
    # The AppTest API allows checking elements within tabs, but verifying
    # the sheer existence of tabs implies the layout loaded properly.
    # Note: Because the API isn't necessarily mocked here, the datasets
    # may fail to fetch, but the UI should handle that gracefully with a warning.
    
    warnings = [w.value for w in at.warning]
    # It's okay if there's a warning about "No datasets found" if the backend is down,
    # but the app itself should not crash.
    
    # We can also check that the sidebar inputs are rendered
    assert len(at.sidebar.text_input) >= 1
    
    # Verify the specific tab headers exist (if the UI hasn't crashed)
    headers = [h.value for h in at.header]
    assert any("Manage Datasets" in h for h in headers)
    assert any("Test Retrieval Engine" in h for h in headers)
    assert any("Standard RAG Chat" in h for h in headers)
    assert any("Agentic RAG Chat" in h for h in headers)
    assert any("API Analytics" in h for h in headers)
    assert any("Local Chunk Tester" in h for h in headers)
