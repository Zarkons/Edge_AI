import os
import sys

# -- Project Information -----------------------------------------------------
project = 'EdgeAI'
copyright = '2026, Cesljarov'
author = 'Zarko Cesljarov'
version = '1.0'
release = '1.0.0'

# -- General Configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',     # Pulls in Python docstrings
    'sphinx.ext.viewcode',    # Links your docs directly to source code
    'sphinx.ext.napoleon',    # Supports Google/NumPy style docstrings
    'breathe',                # Consumes Doxygen XML for C/C++ API docs
]

templates_path = ['_templates']

exclude_patterns = [
    '**/bazel-*',             # Essential: Prevents Sphinx from traversing Bazel symlinks
    '**/.bazelcache',
]

master_doc = 'index'
root_doc = 'index'
language = 'en'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Breathe / Doxygen Integration Setup -------------------------------------

# 1. Provide a default fallback value for standard initialization
breathe_projects = {"EdgeAI": os.path.abspath("../doxygen_xml/xml")}
breathe_default_project = "EdgeAI"


# 2. Hook into Sphinx initialization to catch the -D flag from Bazel
def resolve_breathe_paths(app, config):
    # Sphinx populates -D flags into config.overrides
    doxygen_xml_dir = config.overrides.get("doxygen_xml_dir", "")

    if not doxygen_xml_dir:
        return

    # Check if Bazel provided an absolute path
    if os.path.isabs(doxygen_xml_dir):
        abs_path = doxygen_xml_dir
    else:
        # Dynamically find the root of the Bazel execroot workspace (_main/)
        conf_dir = os.path.dirname(os.path.abspath(__file__))
        if "/bazel-out/" in conf_dir:
            execroot_base = conf_dir.split("/bazel-out/")[0]
            abs_path = os.path.normpath(
                os.path.join(execroot_base, doxygen_xml_dir)
            )
        else:
            # Fallback pathing structure if executed outside sandbox
            abs_path = os.path.normpath(
                os.path.join(conf_dir, "..", "..", "..", "..", doxygen_xml_dir)
            )

    # Check for /xml subfolder layout inside the resolved folder
    final_xml_path = abs_path
    if os.path.isfile(os.path.join(abs_path, "xml", "index.xml")):
        final_xml_path = os.path.join(abs_path, "xml")

    # Override the breathe configuration dynamically
    if os.path.exists(os.path.join(final_xml_path, "index.xml")):
        print(f"\n>>> BREATHE SUCCESS: Found index.xml at {final_xml_path}\n")
        config.breathe_projects = {"EdgeAI": final_xml_path}
    else:
        print(
            f"\n>>> BREATHE ERROR: index.xml not found at target {final_xml_path}\n"
        )


# 3. Connect the path solver to Sphinx's early configuration event
def setup(app):
    app.connect("config-inited", resolve_breathe_paths)


breathe_default_project = 'EdgeAI'


# -- Options for HTML Output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = []
html_show_sphinx = False
