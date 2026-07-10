import os
import sys

# Make local docs modules importable for autodoc in both local and Bazel runs.
CONF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CONF_DIR)
sys.path.insert(0, os.path.join(CONF_DIR, "src"))

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
    'sphinxcontrib.plantuml',
    'sphinx_needs',           # Adds support for requirement tracking in docs
    'sphinx.ext.graphviz',    # Adds support for Graphviz diagrams
    'sphinxcontrib.mermaid',  # Adds support for Mermaid diagrams
]
needs_flow_engine = "graphviz"
needs_id_regex = "^[a-zA-Z0-9_]{5,}"
needs_extra_options = [
    "priority",
    "metric",
    "target",
    "characteristic",
    "req_type",
]
needs_choices = {
    "req_type": ["functional", "quality", "constraint"]
}
needs_extra_links = [
    {
        "option": "achieves",
        "incoming": "achieved_by",
        "outgoing": "achieves",
        "style": "#00FF00",
    },
    {
        "option": "satisfies",
        "incoming": "satisfied_by",
        "outgoing": "satisfies",
        "style": "#00FF00",
    },
    {
        "option": "verifies",
        "incoming": "verified_by",
        "outgoing": "verifies",
        "style": "#0000FF",
    }
]
needs_types = [
    {
        "directive": "goal",
        "title": "Goal",
        "prefix": "goal_",
        "color": "#1f77b4",
        "style": "node",
    },
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "req_",
        "color": "#ff7f0e",
        "style": "node",
    },
    {
        "directive": "spec",
        "title": "Specification",
        "prefix": "spec_",
        "color": "#2ca02c",
        "style": "node",
    },
    {
        "directive": "dec",
        "title": "Decision",
        "prefix": "dec_",
        "color": "#d62728",
        "style": "node",
    },
]
plantuml_ephemeral_with_zooming = True
plantuml_output_format = 'svg'
plantuml_ephemeral_server = 'http://plantuml.com'

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


autodoc_mock_imports = [
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.keras.layers",
    "tensorflow.keras.models",
    "absl",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "seaborn",
    "IPython",
    "IPython.display"
]

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

breathe_default_project = 'EdgeAI'


# -- Options for HTML Output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = []
html_show_sphinx = False


# -- Mermaid Diagram Customizations -----------------------------------------
# Tuning knobs for Mermaid sizing.
# Reduce these values if diagrams look too large.
DOC_BODY_MAX_WIDTH_PX = 1200
MERMAID_MIN_WIDTH_PX = 400
MERMAID_MIN_HEIGHT_PX = 460
MERMAID_NODE_SPACING = 45
MERMAID_RANK_SPACING = 55

# Configures the client-side JavaScript engine running in the user's browser
mermaid_init_js = (
    "mermaid.initialize({\n"
    "    theme: 'default',\n"
    "    startOnLoad: true,\n"
    "    flowchart: {\n"
    "        htmlLabels: true,\n"
    "        useMaxWidth: true,\n"
    f"        nodeSpacing: {MERMAID_NODE_SPACING},\n"
    f"        rankSpacing: {MERMAID_RANK_SPACING},\n"
    "    },\n"
    "    maxTextSize: 100000\n"
    "});\n"
)

# Add this code block at the bottom of conf.py to force larger layouts
def inject_mermaid_styles(app, pagename, templatename, context, doctree):
    # Only inject if a body exists in the context
    if "body" in context:
        context["body"] += (
            """
        <style>
            /* Give content and diagrams more horizontal room in Alabaster. */
            div.bodywrapper {
                max-width: 100% !important;
            }
            div.body {
            """
            + f"                max-width: {DOC_BODY_MAX_WIDTH_PX}px !important;\n"
            + """
                width: 100% !important;
            }

            /* Override sphinxcontrib-mermaid defaults for bigger diagrams. */
            pre.mermaid,
            .mermaid-container > pre {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                margin: 0.5rem 0 !important;
            }

            pre.mermaid > svg,
            .mermaid-container > pre > svg,
            .mermaid svg {
                width: 100% !important;
            """
            + f"                min-width: {MERMAID_MIN_WIDTH_PX}px !important;\n"
            + f"                min-height: {MERMAID_MIN_HEIGHT_PX}px !important;\n"
            + """
                height: auto !important;
                max-width: none !important;
                display: block !important;
                margin: 0 auto !important;
            }
        </style>
        """
        )

def setup(app):
    # Appends styling directly into the HTML output head node
    app.connect("html-page-context", inject_mermaid_styles)

    if hasattr(resolve_breathe_paths, '__call__'):
        app.connect("config-inited", resolve_breathe_paths)
