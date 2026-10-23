"""Launch the HarvestGuard dashboard with: streamlit run dashboard.py."""

from pathlib import Path
import runpy


runpy.run_path(str(Path(__file__).resolve().with_name("harvestguard_app.py")), run_name="__main__")
