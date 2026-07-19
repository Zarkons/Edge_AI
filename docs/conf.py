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
    "verification_method",
    "verification_criteria",
    "argumentation",
]

needs_extra_options_choices = {
    "verification_method": ["Test", "Review", "N/A", "Compile-Time Check", "Static Analysis"],
}

needs_extra_links = [
    {
        # Strategy executes Requirement
        "option": "executes",
        "incoming": "executed_by",
        "outgoing": "executes",
        "style": "#FF0000"
    },
    {
        # Strategy satisfies Goal
        "option": "satisfies",
        "incoming": "satisfied_by",
        "outgoing": "satisfies",
        "style": "#00FF00",
    },
    {
        # Strategy adheres to Constraint
        "option": "adheres_to",
        "incoming": "adhered_by",
        "outgoing": "adheres_to",
        "style": "#A000FF",
    },
    {
        # Decision enables Strategy
        "option": "enables",
        "incoming": "enabled_by",
        "outgoing": "enables",
        "style": "#FF00FF",
    },
    {
        # Concept realizes Decision
        "option": "realizes",
        "incoming": "realized_by",
        "outgoing": "realizes",
        "style": "#FFA500",
    },
    {
        # Specification implements Concept
        "option": "implements",
        "incoming": "implemented_by",
        "outgoing": "implements",
        "style": "#800080",
    },
    {
        # Test verifies Requirement, Goal, Constraint, or Specification
        "option": "verifies",
        "incoming": "verified_by",
        "outgoing": "verifies",
        "style": "#0000FF",
    },
    {
        # Requirement, Goal, Constraint, or Specification is assigned to a Component
        "option": "assigned_to",
        "incoming": "assigned_by",
        "outgoing": "assigned_to",
        "style": "#0000FF",
    },
    {
        # Component is decomposed into sub-components
        "option": "decomposed_from",
        "incoming": "decomposed_by",
        "outgoing": "decomposed_from",
        "style": "#0000FF",
    },
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
        "directive": "adr",
        "title": "Decision",
        "prefix": "adr_",
        "color": "#d62728",
        "style": "node",
    },
    {
        "directive": "strategy",
        "title": "Strategy",
        "prefix": "str_",
        "color": "#9467bd",
        "style": "node",
    },
    {
        "directive": "concept",
        "title": "Concept",
        "prefix": "con_",
        "color": "#8c564b",
        "style": "node",
    },
    {
        "directive": "constraint",
        "title": "Constraint",
        "prefix": "const_",
        "color": "#e377c2",
        "style": "node",
    },
    {
        "directive": "component",
        "title": "Component",
        "prefix": "comp_",
        "color": "#7f7f7f",
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
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_show_sphinx = False

html_theme_options = {
    # Keeps your entire recursive audio/image recognition menu structure open on the side
    "show_navbar_depth": 3,
    "collapse_navigation": False,

    # Directly locks the theme to a permanent light mode style
    "theme_dev_mode": False,
    "default_mode": "light",
}


# -- Mermaid Diagram Customizations -----------------------------------------
# Tuning knobs for Mermaid sizing.
MERMAID_NODE_SPACING = 45
MERMAID_RANK_SPACING = 55

# Configures the client-side JavaScript engine running in the user's browser
mermaid_init_js = (
    "mermaid.initialize({\n"
    "    theme: 'default',\n"
    "    startOnLoad: true,\n"
    "    flowchart: {\n"
    "        htmlLabels: true,\n"
    "        useMaxWidth: false,\n"
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
                width: 100% !important;
            }

            /* Keep Mermaid diagrams at natural size and scroll when needed. */
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
                height: auto !important;
                max-width: none !important;
                width: auto !important;
                display: block !important;
                margin: 0 auto !important;
            }

            .mermaid-zoom-controls {
                display: flex;
                gap: 0.35rem;
                justify-content: flex-end;
                align-items: center;
                margin: 0.25rem 0 0.35rem;
                font-size: 0.85rem;
            }

            .mermaid-zoom-controls button {
                border: 1px solid #bbb;
                border-radius: 0.35rem;
                background: #fff;
                color: #333;
                padding: 0.2rem 0.55rem;
                cursor: pointer;
            }

            .mermaid-zoom-controls button:hover {
                background: #f3f3f3;
            }

            .mermaid-zoom-controls .mermaid-zoom-label {
                min-width: 3.5rem;
                text-align: right;
                color: #666;
                user-select: none;
            }
        </style>

        <script>
        (function () {
            var ZOOM_STEP = 0.25;
            var ZOOM_MIN = 0.5;
            var ZOOM_MAX = 3.0;
            var DEFAULT_BASELINE_SCALE = 1.3;

            function getViewBoxSize(svg) {
                var viewBox = svg.getAttribute('viewBox');
                if (!viewBox) {
                    return null;
                }

                var parts = viewBox.trim().split(/[\s,]+/).map(parseFloat);
                if (parts.length !== 4 || isNaN(parts[2]) || isNaN(parts[3])) {
                    return null;
                }

                return { width: parts[2], height: parts[3] };
            }

            function clamp(value, min, max) {
                return Math.min(max, Math.max(min, value));
            }

            function measureSvg(svg) {
                var width = parseFloat(svg.getAttribute('data-mermaid-base-width'));
                var height = parseFloat(svg.getAttribute('data-mermaid-base-height'));
                var viewBoxSize = getViewBoxSize(svg);

                if (viewBoxSize && (
                    !width || !height ||
                    (width <= 300 && height <= 150) ||
                    Math.abs(width - viewBoxSize.width) > 1 ||
                    Math.abs(height - viewBoxSize.height) > 1
                )) {
                    width = viewBoxSize.width;
                    height = viewBoxSize.height;
                }

                if (!width || !height) {
                    if (!width || !height) {
                        var rect = svg.getBoundingClientRect();
                        width = rect.width || parseFloat(svg.getAttribute('width')) || 800;
                        height = rect.height || parseFloat(svg.getAttribute('height')) || 400;
                    }

                    svg.setAttribute('data-mermaid-base-width', String(width));
                    svg.setAttribute('data-mermaid-base-height', String(height));
                }

                return { width: width, height: height };
            }

            function getFitZoom(shell, svg) {
                var base = measureSvg(svg);
                var availableWidth = shell.clientWidth || base.width;
                var availableHeight = Math.floor(window.innerHeight * 0.8);
                var widthZoom = availableWidth / base.width;
                var heightZoom = availableHeight / base.height;

                return Math.min(widthZoom, heightZoom, 1);
            }

            function applyZoom(content, svg, shell, zoom) {
                var base = measureSvg(svg);
                var fitZoom = getFitZoom(shell, svg);
                var nextZoom = clamp(zoom, ZOOM_MIN, ZOOM_MAX);
                var actualZoom = fitZoom * DEFAULT_BASELINE_SCALE * nextZoom;

                content.style.width = (base.width * actualZoom) + 'px';
                content.style.height = (base.height * actualZoom) + 'px';
                svg.setAttribute('data-mermaid-zoom', String(nextZoom));
                return nextZoom;
            }

            function stabilizeZoom(content, svg, shell, getZoom, updateLabel) {
                [0, 100, 300, 700].forEach(function (delay) {
                    window.setTimeout(function () {
                        var zoom = applyZoom(content, svg, shell, getZoom());
                        updateLabel(zoom);
                    }, delay);
                });
            }

            function addControls(target) {
                if (!target || target.getAttribute('data-mermaid-zoom-ready') === 'true') {
                    return;
                }

                var svg = target.querySelector('svg');
                if (!svg) {
                    return;
                }

                var shell = document.createElement('div');
                shell.className = 'mermaid-zoom-shell';
                shell.style.overflow = 'auto';
                shell.style.maxWidth = '100%';
                shell.style.maxHeight = '80vh';
                shell.style.width = '100%';
                shell.style.margin = '0.5rem 0';

                var content = document.createElement('div');
                content.className = 'mermaid-zoom-content';
                content.style.position = 'relative';
                content.style.display = 'block';

                target.parentNode.insertBefore(shell, target);
                shell.appendChild(content);
                content.appendChild(target);

                target.setAttribute('data-mermaid-zoom-ready', 'true');
                target.style.width = '100%';
                target.style.height = '100%';
                target.style.maxWidth = 'none';
                target.style.display = 'block';
                svg.style.width = '100%';
                svg.style.height = '100%';
                svg.style.maxWidth = 'none';
                svg.style.display = 'block';

                var controls = document.createElement('div');
                controls.className = 'mermaid-zoom-controls';
                controls.innerHTML = '<button type="button" data-mermaid-action="out">-</button>' +
                    '<button type="button" data-mermaid-action="reset">100%</button>' +
                    '<button type="button" data-mermaid-action="in">+</button>' +
                    '<span class="mermaid-zoom-label">100%</span>';

                shell.parentNode.insertBefore(controls, shell);

                var label = controls.querySelector('.mermaid-zoom-label');
                var zoom = applyZoom(content, svg, shell, 1);

                function updateLabel(value) {
                    label.textContent = Math.round(value * 100) + '%';
                }

                updateLabel(zoom);
                stabilizeZoom(content, svg, shell, function () { return zoom; }, updateLabel);

                window.addEventListener('resize', function () {
                    zoom = applyZoom(content, svg, shell, zoom);
                    updateLabel(zoom);
                });

                controls.addEventListener('click', function (event) {
                    var action = event.target && event.target.getAttribute('data-mermaid-action');
                    if (!action) {
                        return;
                    }

                    if (action === 'reset') {
                        zoom = applyZoom(content, svg, shell, 1);
                    } else if (action === 'in') {
                        zoom = applyZoom(content, svg, shell, zoom + ZOOM_STEP);
                    } else if (action === 'out') {
                        zoom = applyZoom(content, svg, shell, zoom - ZOOM_STEP);
                    }

                    updateLabel(zoom);
                });
            }

            function scanMermaidDiagrams() {
                document.querySelectorAll('pre.mermaid, .mermaid-container > pre, .mermaid').forEach(addControls);
            }

            function observeMermaidDiagrams() {
                var scheduled = false;

                function requestScan() {
                    if (scheduled) {
                        return;
                    }

                    scheduled = true;
                    window.requestAnimationFrame(function () {
                        scheduled = false;
                        scanMermaidDiagrams();
                    });
                }

                scanMermaidDiagrams();

                if (document.body) {
                    var observer = new MutationObserver(function () {
                        requestScan();
                    });

                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });

                    window.addEventListener('beforeunload', function () {
                        observer.disconnect();
                    });
                }
            }

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', observeMermaidDiagrams);
            } else {
                observeMermaidDiagrams();
            }
        }());
        </script>
        """
        )

def setup(app):
    # Appends styling directly into the HTML output head node
    app.connect("html-page-context", inject_mermaid_styles)

    if hasattr(resolve_breathe_paths, '__call__'):
        app.connect("config-inited", resolve_breathe_paths)
