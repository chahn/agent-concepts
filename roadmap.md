# Reflective Agent Architecture with Self-Synthesizing Tool Graphs: Toward Composable, LLM-Augmented Workflow Construction and Procedural Memory

---

## Abstract

We propose a **reflective agent architecture** built on a hybrid ReAct/CodeAct pattern that extends the conventional tool-use paradigm by enabling the agent to *author, persist, and compose its own executable code blocks as first-class tools*. Unlike standard tool-calling agents — which operate over a fixed, developer-defined toolset — our agent dynamically extends its capabilities by writing code that can be saved for deferred execution, reused across sessions, and composed into multi-step workflows. Each code block may itself invoke large language models (LLMs) for sub-tasks such as classification, summarization, translation, or structured data extraction, effectively enabling the agent to construct *sub-agents* and *AI-augmented pipelines* at runtime.

A central claim of this work is that persisted code blocks, composed workflows, and constructed graphs constitute a form of **procedural memory** for the agent — an externalized, inspectable, and evolving repository of *how to do things*. Where conventional agent memory stores facts (declarative) or conversation history (episodic), our architecture enables the agent to accumulate and refine **skills as executable knowledge**. The agent does not merely remember that a task was completed; it retains the operational procedure for completing it, can retrieve and re-execute it, and can compose previously acquired procedures into novel higher-order workflows. This makes the architecture a vehicle for *cumulative skill acquisition* in agentic systems. Importantly, the resulting workflow graphs follow what we call the **"Build with AI, Execute Deterministically"** paradigm: the AI agent designs and wires the graph, but the resulting pipeline can be triggered and run by users, schedulers, or APIs with zero LLM involvement and zero hallucination risk at runtime — provided the constituent nodes are standard code. The LLM is a *design-time cost*, not a *runtime cost*.

This self-extending property raises fundamental questions at the intersection of agent architecture, dataflow programming, and graph-based workflow composition. In particular, we identify a core tension in **state management for dynamically composed heterogeneous processing graphs**: code blocks require interoperability (suggesting universal input/output contracts), yet the intermediate outputs between sequential steps are often ephemeral, schema-diverse, and ill-suited to storage in a monolithic shared state. We argue that this problem demands a *layered state model* that distinguishes between edge-scoped data flow, workflow-scoped context, and persistent agent memory — drawing on concepts from dataflow programming, scoped environments in programming language theory, and hypergraph-based workflow representations.

We identify the **Agent Skills** open standard (agentskills.io) — a portable format for packaging agent capabilities as folders of instructions, scripts, and resources — as a natural serialization layer for procedural memory in this architecture. While not yet integrated into the smolagents framework our prototype is built on, Agent Skills provides the discovery, activation, and progressive disclosure mechanisms that map directly onto the procedural memory lifecycle. We propose extensions to the standard to support composability metadata and workflow graph packaging.

We outline a research agenda organized around five pillars: (1) formalizing composability contracts for agent-authored code blocks, (2) designing a layered state architecture for dynamic workflow graphs, (3) developing graph-theoretic representations for self-modifying agent workflows, (4) understanding procedural memory formation, retrieval, and refinement in self-extending agents, and (5) building evaluation frameworks for correctness, safety, and emergent capability.

---

## 1. Motivation and Problem Statement

### 1.1 Beyond Static Tool Sets

Current agentic frameworks generally assume a **static tool inventory**: developers define tools, and the agent selects and sequences them at runtime. This creates a ceiling — the agent can only do what its tools allow. Our architecture removes this ceiling by treating code authorship itself as a tool:

- The agent can **write** executable code blocks (Python functions, scripts).
- These blocks can be **persisted** with metadata (name, description, execution triggers).
- They can be **invoked** by the agent or by the user, immediately or at a later point.
- They can be **composed** into multi-step pipelines or directed acyclic graphs (DAGs).

This makes the agent *reflective*: it can inspect, modify, and extend its own operational capabilities.

### 1.2 Code, Graphs, and Workflows as Procedural Memory

A fundamental observation: when the agent writes a code block and persists it, it is not merely producing an artifact — it is **encoding a skill**. When it composes multiple blocks into a workflow graph, it is encoding a *complex procedure*. The totality of an agent's saved code blocks and workflow graphs constitutes its **procedural memory**: a structured, executable, and evolving body of knowledge about *how to accomplish tasks*.

This stands in contrast to other forms of agent memory:

| Memory Type | What It Stores | Example | Representation |
|---|---|---|---|
| **Declarative** | Facts, data, beliefs | "The user prefers PDF output" | Key-value store, knowledge graph |
| **Episodic** | Event history, past interactions | "Last time, the user asked for a quarterly report" | Conversation logs, retrieval-augmented memory |
| **Procedural** | Skills, operations, how-to knowledge | "To create a bilingual report: extract → translate → format → merge → export" | Persisted code blocks, workflow graphs, composed pipelines |

Procedural memory in our architecture has several distinctive properties:

- **Executable.** Unlike declarative memories that must be interpreted, procedural memories can be directly executed. A saved workflow is not a description of how to do something — it *is* the doing.
- **Inspectable and modifiable.** Because procedures are represented as code and graphs (not opaque neural weights), the agent — and the human — can inspect, debug, and refine them. This is transparent skill acquisition.
- **Composable.** Existing procedural memories (code blocks) serve as building blocks for new, more complex procedures (workflow graphs). This enables *hierarchical skill composition*: simple skills combine into compound skills, which combine into sophisticated multi-stage workflows.
- **Transferable.** Procedural memories can be shared between agent instances, exported, version-controlled, and collaboratively developed — unlike implicit knowledge stored in model weights.
- **AI-optional at runtime ("Build with AI, Execute Deterministically").** A critical design property: once a workflow graph has been constructed, its execution does not inherently require an AI agent or LLM. If all nodes in the graph are deterministic code blocks (no LLM-in-the-loop), the workflow is a purely conventional data processing pipeline — executable manually by a user, triggered by a scheduler or external event, or embedded in a traditional software system. The agent is the *architect* of the workflow, but need not be the *executor*. This means the architecture produces artifacts that are valuable independently of the AI system that created them: a workflow authored by the agent today can run as a standalone automation tomorrow, with no LLM costs and no hallucination risk at runtime.
- **Self-improving.** The agent can revisit and refine its own procedures over time, replacing naive implementations with optimized ones, adding error handling, or adapting procedures to new requirements.

This framing connects deeply to classical AI concepts of procedural knowledge representation, but with a crucial modern twist: the *LLM is both the author and the consumer of its own procedural memory*. The agent writes code it will later execute, designs workflows it will later orchestrate, and builds tools it will later use. This feedback loop — from experience to encoded skill to improved execution — is the core mechanism for cumulative capability growth in our architecture.

### 1.3 LLM-in-the-Loop Code Blocks

A critical feature is that authored code blocks are not limited to deterministic computation. A code block can invoke an LLM to perform:

- **Classification** (e.g., routing documents to processing pipelines)
- **Summarization** (e.g., condensing extracted text before further processing)
- **Translation** (e.g., language normalization in multilingual pipelines)
- **Structured extraction** (e.g., extracting tabular data from PDFs or images)
- **Decision-making** (e.g., evaluating whether intermediate output meets quality criteria)

This means code blocks are not merely scripts — they are *hybrid computational units* that blend deterministic logic with probabilistic LLM reasoning. When composed, they form **AI-augmented dataflow graphs** where some nodes are pure functions, some are LLM calls, and some are themselves small agents.

From the procedural memory perspective, this is significant: the agent's skills are not limited to conventional programming. The agent can encode *judgment-requiring procedures* — tasks that involve interpretation, ambiguity resolution, and context-dependent reasoning — as persistent, reusable workflow steps. A "skill" in this architecture can be something like "read an invoice PDF, extract line items into structured data, classify expenses by category, and flag anomalies" — a procedure that fundamentally requires LLM capabilities at multiple stages.

### 1.4 The State Management Problem

When composing code blocks into workflows, we face a fundamental design tension:

| Approach | Advantage | Disadvantage |
|---|---|---|
| **Universal I/O contract** (every block takes `Dict` → returns `Dict`) | Simple composition, interchangeable blocks | Loss of type safety; semantic mismatch between blocks; forces all data into flat key-value structure |
| **Shared global state** (all blocks read/write a common object) | Blocks can access any prior result | Tight coupling; namespace collisions; difficult to reason about data dependencies; no clear lifecycle for intermediate data |
| **Typed edge connections** (block outputs are typed and explicitly wired to inputs) | Clean dependencies; type safety | Rigid; hard for agents to compose dynamically; schema evolution is expensive |

The specific challenge: **ephemeral intermediate outputs**. Step *n* produces data consumed by step *n+1* but irrelevant thereafter. Storing this in a persistent shared state object pollutes the namespace, creates ambiguity about data lifecycle, and makes workflow graphs harder to analyze. Yet a purely edge-based model (typed connections) is too rigid for an agent that dynamically constructs its own workflows.

---

## 2. Proposed Architecture

### 2.1 Layered State Model

We propose a **four-layer state hierarchy** inspired by scoping rules in programming languages and the blackboard architecture pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Persistent Agent Memory (incl. Procedural Memory)     │
│  (learned tools, workflows, user prefs, long-term knowledge)    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Workflow Context                                      │
│  (shared state for a single workflow execution)                 │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Edge / Step-Scoped Data                               │
│  (ephemeral: output of step n, input to n+1)                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Block-Local State                                     │
│  (internal variables within a single code block)                │
└─────────────────────────────────────────────────────────────────┘
```

**Layer 1 — Block-Local State.** Internal to a single code block. Not visible to other blocks. Analogous to local variables in a function.

**Layer 2 — Edge-Scoped / Ephemeral State.** The return value of one block, passed as input to the next. Has a defined lifecycle (created at step *n*, consumed at step *n+1*, garbage-collected thereafter). This is the *data on the edge* in a dataflow graph. It resolves the "temporary output" problem: the data exists only for the duration of the edge traversal.

**Layer 3 — Workflow Context.** A scoped, mutable context object shared across all blocks within a single workflow execution. Appropriate for cross-cutting concerns: configuration, accumulated metadata, logging, provenance tracking. Analogous to a thread-local context or a request-scoped dependency injection container.

**Layer 4 — Persistent Agent Memory.** Survives across workflow executions and sessions. This layer *includes the procedural memory* — the agent's library of authored code blocks and composed workflow graphs — alongside declarative knowledge (user preferences, domain facts) and episodic traces (execution history, success/failure records). The procedural memory component is what makes this architecture self-extending: every successfully completed workflow can contribute a new skill to the agent's permanent repertoire.

The key insight: **ephemeral intermediate outputs live at Layer 2, not Layer 3.** They flow along edges in the workflow graph and are naturally scoped to the step transition. The workflow context (Layer 3) is reserved for data that genuinely needs to be accessible by multiple non-adjacent steps.

### 2.2 Code Block Contract

Each code block conforms to a minimal interface:

```python
@dataclass
class BlockSignature:
    name: str
    description: str               # Natural-language; used by the agent for selection
    input_schema: JSONSchema        # What this block expects (Layer 2 input)
    output_schema: JSONSchema       # What this block produces (Layer 2 output)
    context_reads: list[str]        # Keys this block reads from workflow context (Layer 3)
    context_writes: list[str]       # Keys this block writes to workflow context (Layer 3)
    requires_llm: bool              # Whether this block invokes an LLM
    triggers: list[TriggerSpec]     # Conditions under which this block should be invoked
    version: str                    # Enables procedural memory evolution
    lineage: list[str]              # Chain of blocks this was derived from
```

The `version` and `lineage` fields support procedural memory management: the agent can track how skills evolve over time and maintain provenance chains when one procedure is refined into another.

### 2.3 Workflow as Graph

A composed workflow is a **directed acyclic graph (DAG)** — or, in the presence of feedback loops and conditional branching, a **directed graph with control-flow annotations**:

- **Nodes** are code blocks (with their `BlockSignature`).
- **Edges** carry typed data (Layer 2 state).
- **The workflow context** (Layer 3) is a shared annotation on the graph, accessible to all nodes.
- **Control flow** (branching, looping, error handling) is expressed through annotated edges or dedicated control-flow nodes.

Critically, the graph itself is a *first-class object in the agent's procedural memory*. It can be saved, named, versioned, and reused. A workflow graph is not just an execution plan — it is a **learned compound skill**.

### 2.4 "Build with AI, Execute Deterministically"

A fundamental architectural principle — and a core selling point for both academic and industrial adoption — is what we call the **"Build with AI, Execute Deterministically"** paradigm: the AI agent designs and wires the workflow graph, but the resulting pipeline can run with zero LLM involvement at runtime.

The execution engine that traverses the graph, manages state across layers, and invokes code blocks is a conventional software component — not an LLM. Once the agent has figured out how to solve a problem and composed the graph, a human, a CRON job, or an API endpoint can press "play" and run that exact pipeline with **zero LLM API costs and zero hallucination risk** — provided the underlying nodes are standard code.

This has several important implications:

- **Manual triggering.** A user can trigger a saved workflow directly, without involving the agent at all. The graph executes deterministically, taking input and producing output like any data processing pipeline.
- **Scheduled / event-driven execution.** Workflows can be attached to external triggers — cron schedules, file system events, API webhooks, database changes — and run unattended. The agent authored the workflow; the scheduler runs it.
- **Fully AI-free workflows.** If all nodes in a graph are pure code blocks (no LLM calls), the entire workflow operates without any AI involvement whatsoever. It is a conventional pipeline that happens to have been *designed* by an AI agent. The architecture produces durable, inspectable, deterministic automation artifacts as a byproduct of agent interaction.
- **Hybrid workflows.** In practice, many workflows will be hybrid: some nodes are deterministic (parsing, formatting, calculations, API calls), others invoke LLMs (classification, summarization, extraction). The `requires_llm` flag in `BlockSignature` makes this explicit per node, allowing the runtime — and the user — to understand exactly where AI reasoning enters the pipeline and where it does not.
- **Graceful degradation.** If LLM access is unavailable (cost constraints, latency requirements, offline environments), the runtime can identify which nodes require it, execute all deterministic nodes, and either skip, queue, or flag the LLM-dependent nodes for later processing.

This paradigm reframes the economic model of AI-assisted automation: the LLM is a *design-time cost*, not a *runtime cost*. A team uses the agent to *design* a complex data processing workflow once, then runs that workflow thousands of times as a conventional automation — reproducible, auditable, and free of stochastic variance. The AI investment is amortized across all subsequent executions.

For academia, this raises a novel class of questions around the boundary between AI-designed and AI-executed computation: what fraction of real-world workflows can be fully "compiled away" from LLM dependence? What are the formal properties of the deterministic sub-graph vs. the LLM-dependent sub-graph in a hybrid workflow? For industry, the value proposition is immediate: AI as a one-time engineering cost for automation design, not a recurring operational cost for automation execution.

### 2.5 Procedural Memory Lifecycle

The procedural memory evolves through a cycle:

1. **Acquisition.** The agent encounters a novel task, decomposes it, and authors new code blocks and/or a workflow graph to solve it.
2. **Execution.** The workflow runs — triggered by the agent, by a user manually, or by an external event. Results are observed.
3. **Persistence.** If successful (or partially successful), the code blocks and workflow graph are saved to Layer 4 with metadata about the task context and execution outcome.
4. **Retrieval.** When a similar task arises, the agent searches its procedural memory for relevant blocks and workflow patterns.
5. **Composition.** Retrieved blocks are combined — possibly with newly authored blocks — into a workflow for the new task. Existing workflows may be adapted rather than rebuilt from scratch.
6. **Refinement.** Based on execution feedback, the agent may modify existing blocks (fix bugs, optimize, generalize) or restructure workflow graphs (add error handling, parallelize steps, insert validation stages).

This cycle mirrors how human procedural knowledge develops: initial effortful construction, followed by retrieval and adaptation, with gradual refinement toward efficiency and robustness.

### 2.6 Agent Skills as the Packaging Standard for Procedural Memory

A crucial practical dimension of this architecture is the question of *how procedural memory is serialized, shared, and discovered*. Here, we identify a strong alignment with the **Agent Skills** open standard (agentskills.io) — a lightweight, portable format for packaging agent capabilities as self-contained directory structures.

An Agent Skill is a folder containing a `SKILL.md` file (metadata + natural-language instructions), optional executable scripts, reference documentation, and static assets:

```
skill-name/
├── SKILL.md          # Metadata (name, description) + instructions
├── scripts/          # Executable code (Python, Bash, JS)
├── references/       # Documentation, technical details
└── assets/           # Templates, schemas, resources
```

The format uses **progressive disclosure** to manage context efficiently: at startup, agents load only skill names and descriptions (~100 tokens each); when a task matches, the full `SKILL.md` body is loaded; scripts and references are loaded only when actively needed during execution. This three-tier loading model directly mirrors the distinction between skill *discovery*, *activation*, and *execution* in our procedural memory lifecycle (Section 2.4).

The alignment between Agent Skills and our architecture is deep:

| Our Concept | Agent Skills Realization |
|---|---|
| Persisted code block | `scripts/` directory containing executable code |
| Block metadata (`BlockSignature`) | SKILL.md frontmatter (`name`, `description`, `compatibility`) |
| Natural-language description for agent selection | `description` field, optimized for agent discovery |
| Procedural instructions | SKILL.md Markdown body |
| Supporting resources (schemas, templates) | `assets/` and `references/` directories |
| Version tracking | `metadata.version` field |
| Composability information | Could be extended via `metadata` or additional frontmatter fields |

Agent Skills thus provides a **ready-made serialization format for Layer 4 procedural memory**. When our agent authors a new code block and persists it, it can package it as an Agent Skill. When it retrieves a skill from its library, it follows the progressive disclosure pattern. When skills are shared between agent instances or teams, the format guarantees portability.

However, the current Agent Skills specification does not natively address several concerns central to our architecture:

- **Composability metadata.** There is no standard way to declare input/output schemas for a skill, making automated schema-matching between skills impossible without extension.
- **Workflow graphs.** Skills are individual units; there is no standard for packaging a *composed workflow* (a graph of skills) as a higher-order skill.
- **Edge-scoped state.** The specification does not address how data flows between skills when they are composed.
- **LLM-in-the-loop declaration.** There is no way to indicate that a skill requires LLM access or what kind of LLM reasoning it performs.

These gaps represent concrete **extension opportunities**: our research can propose additions to the Agent Skills specification that support composability, workflow packaging, and the layered state model. This positions the project not only as a consumer of the standard but as a contributor to its evolution.

Critically, Agent Skills support is **not yet implemented in smolagents** — the framework our prototype is built on. Implementing this integration is a near-term deliverable that would simultaneously advance the research agenda and produce a practical, reusable open-source contribution.

---

## 3. Connections to Existing Frameworks and Key Distinctions

### 3.1 Relation to LangChain / LangGraph

| Concept | LangChain/LangGraph | Our Architecture |
|---|---|---|
| Tool definition | Developer-authored, static | Agent-authored, dynamic, persistent |
| Workflow | Developer-defined graph or chain | Agent-constructed graph at runtime |
| State management | Typed state channels | Layered model (edge-scoped + workflow context + persistent) |
| Sub-agents | Predefined agent nodes | Dynamically created via LLM-in-the-loop code blocks |
| Self-modification | Not supported | Core capability |
| Memory of procedures | Not explicit | First-class: code blocks and graphs *are* procedural memory |
| Skill packaging standard | Not applicable | Agent Skills format (agentskills.io) as serialization layer |
| Runtime execution model | LLM required at every invocation | "Build with AI, Execute Deterministically" — LLM at design time, optional at runtime |

### 3.2 Key Theoretical Touchpoints

The architecture connects to several established areas — each representing a potential axis of formal inquiry:

- **Dataflow Programming.** Our workflow graphs are dataflow graphs. The edge-scoped state model corresponds to tokens on arcs in a dataflow network. The distinction between data edges (Layer 2) and shared context (Layer 3) maps to the distinction between dataflow and control flow.

- **Blackboard Architecture.** The workflow context (Layer 3) is a scoped blackboard. But unlike classic blackboard systems, our architecture provides layered scoping rather than a single shared space.

- **Reflective Architectures.** The agent's ability to author and modify its own tools is a form of computational reflection — the system can inspect and alter its own structure.

- **Graph Rewriting Systems.** When the agent modifies a workflow by adding, removing, or reconnecting code blocks, it performs graph rewriting. Formalizing the agent's modification capabilities as a graph grammar could enable reasoning about the space of reachable workflow configurations.

- **Hypergraph Representations.** Code blocks with multiple inputs and outputs are naturally represented as hyperedges. Workflow composition then becomes hypergraph construction, enabling richer representations than simple directed graphs.

- **Program Synthesis.** The agent's construction of new code blocks from natural-language intent is a form of LLM-guided program synthesis. The composition of blocks into workflows is synthesis at a higher level of abstraction.

- **Procedural Knowledge Representation.** The idea that an agent's skills can be represented as inspectable, executable, composable structures connects to long-standing work on procedural knowledge in both AI and cognitive science. Our contribution is the specific representation choice — code blocks + graph composition + layered state — and the LLM-driven acquisition mechanism.

- **Category Theory / Composability.** Code blocks as morphisms, their input/output types as objects, sequential composition as morphism composition, and parallel composition as a monoidal product. This could provide a formal foundation for reasoning about when two blocks are composable.

---

## 4. Research Agenda and Roadmap

### Phase 1: Formal Foundations

**Objective:** Establish the theoretical framework and core abstractions.

- **Task 1.1** — Formalize the layered state model. Define the scoping rules, lifecycle semantics, and garbage collection for each layer. Specify when data should live at Layer 2 (edge) vs. Layer 3 (context) vs. Layer 4 (persistent).
- **Task 1.2** — Define the `BlockSignature` contract and schema language. Investigate JSON Schema, algebraic data types, or dependent types as candidates. Evaluate the tradeoff between expressiveness and the agent's ability to reason about schemas.
- **Task 1.3** — Formalize workflow graphs. Determine the appropriate graph formalism (DAG, directed graph with annotations, hypergraph) and specify the semantics of nodes, edges, and control-flow constructs.
- **Task 1.4** — Formalize the procedural memory model. Define what constitutes a "skill," how skills are indexed for retrieval, how similarity between task contexts and stored procedures is measured, and how versioning/lineage is tracked.
- **Task 1.5** — Analyze the Agent Skills specification against our requirements. Identify gaps (composability metadata, workflow graph packaging, edge-scoped state, LLM-in-the-loop declaration) and propose formal extensions that align the standard with the layered state model and graph-based workflow composition.

**Deliverable:** A formal specification document and a positioning paper.

### Phase 2: Core Runtime Implementation

**Objective:** Build the execution engine for agent-authored workflow graphs.

- **Task 2.1** — Implement the layered state runtime. Build the Layer 2 (edge-scoped) and Layer 3 (context) state managers with proper lifecycle management.
- **Task 2.2** — Implement the code block registry and procedural memory store. Support authoring, persistence, versioning, retrieval, and discovery of code blocks by the agent.
- **Task 2.3** — Implement Agent Skills support in smolagents. Enable the agent to discover, load, and execute skills packaged in the Agent Skills format. This includes the progressive disclosure pipeline (metadata → instructions → scripts/resources) and the ability for the agent to *author new skills* that conform to the standard.
- **Task 2.4** — Implement the workflow graph executor. Support sequential, parallel, branching, and error-handling execution over DAGs of code blocks.
- **Task 2.5** — Integrate LLM-in-the-loop blocks. Standardize how code blocks invoke LLMs, handle streaming, manage token budgets, and propagate errors.

**Deliverable:** A working runtime with API, capable of executing agent-authored multi-step workflows.

### Phase 3: Agent-Level Graph Construction and Skill Acquisition

**Objective:** Enable the agent to design, modify, and optimize workflow graphs — and to build a growing procedural memory.

- **Task 3.1** — Schema-aware block composition. The agent reasons about `input_schema` / `output_schema` compatibility to propose valid block sequences.
- **Task 3.2** — Graph construction from natural language. Given a complex task description, the agent decomposes it into a workflow graph, selecting existing blocks and authoring new ones as needed.
- **Task 3.3** — Procedural memory retrieval and reuse. The agent searches its skill library when encountering new tasks, adapting prior workflows rather than building from scratch.
- **Task 3.4** — Self-modification and graph rewriting. The agent can modify a running or saved workflow graph: insert, replace, or remove nodes; reroute edges; adjust parameters.
- **Task 3.5** — Workflow templates and patterns. The agent learns common workflow patterns (ETL, map-reduce, fan-out/fan-in, human-in-the-loop) and applies them to new problems.

**Deliverable:** An agent that can construct non-trivial multi-step workflows from high-level task descriptions, drawing on an evolving skill library.

### Phase 4: Evaluation, Safety, and Formal Analysis

**Objective:** Assess correctness, safety, and emergent properties.

- **Task 4.1** — Correctness evaluation. Define benchmarks for workflow correctness: does the composed workflow produce the expected output? How does composition quality degrade with workflow complexity?
- **Task 4.2** — Safety analysis. What happens when the agent creates a workflow with side effects, infinite loops, or resource exhaustion? Define a sandboxing model and constraint language.
- **Task 4.3** — Graph-theoretic analysis. Apply graph metrics (depth, width, connectivity, cyclicity) to agent-authored workflows. Correlate graph structure with task complexity and execution performance.
- **Task 4.4** — Procedural memory dynamics. Do agents develop reusable block libraries over time? How does the skill library grow, specialize, and stabilize? Is there convergence toward canonical workflow patterns for recurring task classes? What is the relationship between library size and agent performance on novel tasks?
- **Task 4.5** — Comparative evaluation. Benchmark against static-toolset agents and developer-authored LangGraph workflows on equivalent tasks. Quantify the advantage (or cost) of self-extension.

**Deliverable:** Evaluation framework, benchmark suite, and an analytical paper on emergent workflow structure and procedural memory dynamics.

---

## 5. Key Research Questions

1. **Composability.** What is the right level of type discipline for agent-authored code blocks? How strict should schema contracts be to enable reliable composition without over-constraining the agent's flexibility?

2. **State scoping.** How should we formalize the boundary between ephemeral edge state and persistent workflow context? Can we derive scoping rules automatically from dataflow analysis of the workflow graph?

3. **Graph representation.** Are DAGs sufficient, or do we need hypergraphs, hierarchical graphs, or graph grammars to capture the full space of agent-authored workflows? What are the implications for analyzability?

4. **Procedural memory retrieval.** How should the agent index and search its skill library? By task description similarity? By input/output schema matching? By structural similarity of workflow graphs? What combination yields the best reuse rates?

5. **Skill abstraction and generalization.** Can the agent generalize specific procedures into reusable templates? For instance, after building three different "extract data from PDF → transform → export to spreadsheet" workflows, can it abstract the common pattern and parameterize the differences?

6. **Self-modification safety.** How do we ensure that an agent's graph-rewriting capabilities do not produce unsafe or degenerate workflows? Can graph grammars provide the right constraint language?

7. **LLM-node semantics.** How should we reason about the behavior of non-deterministic (LLM) nodes in a workflow graph? What guarantees (if any) can we provide about end-to-end workflow behavior when some nodes are stochastic?

8. **Execution independence and the deterministic boundary.** In the "Build with AI, Execute Deterministically" paradigm, what fraction of real-world workflows can be fully "compiled away" from LLM dependence? Can the runtime automatically partition a hybrid workflow into deterministic sub-graphs (executable standalone) and LLM-dependent sub-graphs? How should the system handle graceful degradation when LLM access is unavailable — queue, skip, or substitute with fallback logic?

9. **Emergence and convergence.** Over many tasks, do agents develop stable tool libraries and recurring workflow patterns? Can we characterize these patterns formally? Does the procedural memory exhibit properties analogous to human skill consolidation — increasing efficiency, decreasing variability, and chunking of sub-procedures?

10. **Standardization and interoperability.** How should the Agent Skills specification be extended to support composability metadata, workflow graph packaging, and layered state declarations? What is the minimal set of extensions that enables automated skill composition while preserving the format's simplicity and portability?

---

## 6. Technical Context

The current prototype is built on **smolagents** (`ToolCallingAgent`), using Python as the code-execution substrate. The agent operates with four core tools: `write_code`, `read_code`, `execute_code`, and `write_metadata`. Code blocks are persisted as Python files with JSON metadata sidecars. A key near-term integration target is the **Agent Skills** open standard (agentskills.io), which provides a portable packaging format for skills (instructions + scripts + resources) that is already adopted by several agent products but not yet supported in smolagents. Implementing Agent Skills support would simultaneously provide the serialization layer for procedural memory and create an interoperability bridge to the broader agent ecosystem. The immediate next steps are formalizing the block interface, implementing the layered state model, integrating Agent Skills, and progressing to graph-based workflow composition and procedural memory management.
