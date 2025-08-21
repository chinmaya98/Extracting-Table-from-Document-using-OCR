"""
Budget Table Extractor - Main Application
Minimal entry point for the Streamlit application.
"""
from ui_app import run_app


def main():
    """Main application entry point."""
    try:
        # Run the consolidated UI application
        run_app()
    except Exception as e:
        import streamlit as st
        st.error(f"Application failed to start: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
