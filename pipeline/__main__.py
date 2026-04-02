"""
Entry point for running the pipeline as a module:
    python -m pipeline --csv indiana_reports.csv --limit 3 --generator dalle
"""

from .pipeline import main

main()