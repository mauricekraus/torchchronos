import re
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "torchchronos"


with open(Path(__file__).parents[2] / "torchchronos" / "__init__.py") as init:
    metadata = dict(re.findall('''__([a-z]+)__ = "([^"]+)"''', init.read()))

copyright = f"2023, {metadata['author']}"
author = metadata["author"]


version = metadata["version"].split("-", maxsplit=1)[0]
release = metadata["version"]

# -- General configuration ---------------------------------------------------

primary_domain = "py"

# If this is True, todo and todolist produce output, else they produce nothing.
todo_include_todos = True

language = "en"

# Add any Sphinx extension
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "lightning": ("https://pytorch-lightning.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

nitpicky = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# autodoc_mock_imports = ["rclpy"]


# Change the description of the autodoc_typehints directive
autodoc_typehints = "description"


# -- Options for Napoleon extension ------------------------------------------
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
