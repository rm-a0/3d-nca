import os
import sys

# Repo root on sys.path so `from src.core import ...` resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# -- Project information -----------------------------------------------------
project = '3D NCA Framework'
copyright = '2026, Michal Repcik'
author = 'Michal Repcik'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- autodoc -----------------------------------------------------------------
autodoc_mock_imports = [
    'bpy',
    'pyvista',
    'mathutils',
    'matplotlib',
    'trimesh',
]

autodoc_member_order = 'bysource'
autodoc_typehints = 'both'


# -- Napoleon ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
