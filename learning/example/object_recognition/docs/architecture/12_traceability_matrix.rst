12. Traceability Matrix
=======================

This section provides an automated, bidirectional mapping of the system architecture. It tracks how high-level business drivers and quality goals translate into concrete technical requirements, and how those requirements are verified.

.. note::
   This matrix is updated dynamically by the build system during every compilation cycle. Broken links or orphan items will trigger compilation errors.


Functional Requirements
-----------------------

.. needtable::
    :columns: id, title, status, executed_by, verified_by
    :colwidths: 25, 55, 15, 15, 15
    :sort: id
    :filter: type == "req"


Quality Goals
-------------

.. needtable::
    :columns: id, title, status, satisfied_by, verified_by
    :colwidths: 25, 55, 15, 15, 15
    :sort: id
    :filter: type == "goal"

Constraints
-----------

.. needtable::
    :columns: id, title, status, adhered_by, verified_by
    :colwidths: 25, 55, 15, 15, 15
    :sort: id
    :filter: type == "constraint"

Strategies
----------

.. needtable::
    :columns: id, title, status, executes, satisfies, adheres_to, enabled_by
    :colwidths: 25, 55, 15, 15, 15, 15, 15
    :sort: id
    :filter: type == "strategy"

Concepts
--------

.. needtable::
    :columns: id, title, status, realizes, implemented_by
    :colwidths: 25, 55, 15, 15, 15
    :sort: id
    :filter: type == "concept"