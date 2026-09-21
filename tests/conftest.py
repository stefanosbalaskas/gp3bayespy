import os

# Keep plotting tests deterministic and GUI-independent on local and CI hosts.
os.environ.setdefault("MPLBACKEND", "Agg")
