import os
import sys

# Ensure the package root is on sys.path when running tests directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
