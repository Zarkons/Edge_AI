12. Traceability Matrix
=======================

.. list-table:: Traceable Architecture Elements
   :widths: 20 60
   :header-rows: 1

   * - Traceable Element
     - Description
   * - **Requirement**
     - A statement of a behavior, or capability that a system must fulfill.
   * - **Quality Goal**
     - A quality attribute or non-functional requirement that a system must fulfill.
   * - **Constraint**
     - A restriction or limitation on the design or implementation of a system.
   * - **Strategy**
     - A high-level plan or approach for achieving a requirement, goal, or constraint.
   * - **Decision**
     - A choice made between alternative options for enabling a strategy to be realized.
   * - **Concept**
     - A technical solution that realizes a decision.
   * - **Specification**
     - A detailed description of specified behavior that implements a concept.
   * - **Test**
     - A procedure for verifying that a requirement, goal, constraint, or specification has been fulfilled.

.. list-table:: Traceable Relationships
   :widths: 20 20 20 60
   :header-rows: 1

   * - Traceable Relationship
     - Up-Trace
     - Down-Trace
     - Description
   * - **executes**
     - executed_by
     - executes
     - A strategy executes a requirement. Intent: Indicates that a requirement involves an action, operation, or behavioral process performed by the system. The strategy defines how that runtime behavior runs.
   * - **satisfies**
     - satisfied_by
     - satisfies
     - A strategy satisfies a quality goal. Indicates that a quality goal is an inherent characteristic, capability, or non-functional property achieved by the system. The strategy defines the structural constraints or rules that allow this state of quality to exist.
   * - **adheres_to**
     - adhered_by
     - adheres_to
     - A strategy adheres to a constraint. Indicates that a constraint is a restriction or limitation on the design or implementation of the system. The strategy defines how the system is designed to comply with this restriction.
   * - **enables**
     - enabled_by
     - enables
     - A decision enables a strategy. Indicates that a strategy is realized by a decision. The decision defines the choice made between alternative options for enabling the strategy to be realized.
   * - **realizes**
     - realized_by
     - realizes
     - A concept realizes a decision. Indicates that a decision is implemented by a concept. The concept defines the technical solution that realizes the decision.
   * - **implements**
     - implemented_by
     - implements
     - A specification implements a concept. Indicates that a concept is realized by a specification. The specification defines the detailed description of specified behavior that implements the concept.
   * - **verifies**
     - verified_by
     - verifies
     - A test verifies a requirement, goal, constraint, or specification. Indicates that a requirement, goal, constraint, or specification is validated by a test. The test defines the procedure for verifying that the requirement, goal, constraint, or specification has been fulfilled.
   * - **assigned_to**
     - assigned_by
     - assigned_to
     - A requirement, goal, constraint, or specification is assigned to a component. Indicates that a requirement, goal, constraint, or specification is fulfilled by a component. The component defines the structural or behavioral element of the system that fulfills the requirement, goal, constraint, or specification.
   * - **decomposed_from**
     - decomposed_by
     - decomposed_from
     - A component is decomposed into sub-components. Indicates that a component is composed of one or more sub-components. The sub-components define the structural or behavioral elements of the system that make up the parent component.