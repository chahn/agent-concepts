# Reflective Agent Architecture with Self-Synthesizing Tool Graphs: Toward Composable, LLM-Augmented Workflow Construction and Procedural Memory

## Abstract

We propose a **reflective agent architecture** built on a hybrid ReAct/CodeAct pattern that extends the conventional tool-use paradigm by enabling the agent to *author, persist, and compose its own executable code blocks as first-class tools*. Unlike standard tool-calling agents — which operate over a fixed, developer-defined toolset — our agent dynamically extends its capabilities by writing code that can be saved for deferred execution, reused across sessions, and composed into multi-step workflows. Each code block may itself invoke large language models (LLMs) for sub-tasks such as classification, summarization, translation, or structured data extraction, effectively enabling the agent to construct *sub-agents* and *AI-augmented pipelines* at runtime.

The central scientific contribution lies not in any single component but in their integration: existing frameworks offer either CodeAct without persistence (smolagents), skill persistence without industrial applicability (Voyager), or workflow execution without agent-driven construction (n8n, LangGraph). Our architecture bridges the gap between academic proof-of-concept and industry-ready framework by organizing persisted, versioned, and composable code blocks into a **skill library** — a structured, executable, and growing collection of *operational procedures*. The agent stores not only *that* a task was solved but *how* — and can retrieve, reuse, and compose that solution with others into more complex workflows. Importantly, the resulting workflow graphs follow what we call the **"Build with AI, Execute Deterministically"** paradigm: the AI agent designs and wires the graph, but the resulting pipeline can be triggered and run by users, schedulers, or APIs with zero LLM involvement and zero hallucination risk at runtime — provided the constituent nodes are standard code. The LLM is a *design-time cost*, not a *runtime cost*.

This self-extending property raises fundamental questions at the intersection of agent architecture, dataflow programming, and graph-based workflow composition. In particular, we identify a core tension in **state management for dynamically composed heterogeneous processing graphs**: code blocks require interoperability (suggesting universal input/output contracts), yet the intermediate outputs between sequential steps are often ephemeral, schema-diverse, and ill-suited to storage in a monolithic shared state. We argue that this problem demands a *layered state model* that distinguishes between edge-scoped data flow, workflow-scoped context, and persistent agent memory — drawing on concepts from dataflow programming, scoped environments in programming language theory, and hypergraph-based workflow representations.

The **[Agent Skills](https://agentskills.io/home)** open standard ([specification](https://agentskills.io/specification)) — a portable format for packaging agent capabilities — serves as the natural serialization layer for the skill library. We propose extensions to the standard to support composability metadata and workflow graph packaging. The architecture is deliberately framework-agnostic: smolagents — an open-source framework maintained by Hugging Face — is a modular, replaceable component.

The research agenda is organized around four guiding questions: (1) formalizing composability contracts for agent-authored code blocks within the dataflow programming paradigm, (2) skill retrieval and composition in growing skill libraries, (3) secure credential management for agent-authored code with external API access, and (4) determining the deterministic boundary in hybrid workflows. The project is carried out as a collaborative research effort between the University of Bayreuth and a Bavarian technology company, targeting technology transfer into the Bavarian Mittelstand, the manufacturing industry, and public administration.

---

## 1. Motivation and Problem Statement

### 1.1 Beyond Static Tool Sets

Current agentic frameworks generally assume a **static tool inventory**: developers define tools, and the agent selects and sequences them at runtime. This creates a ceiling — the agent can only do what its tools allow. Our architecture removes this ceiling by treating code authorship itself as a tool:

- The agent can **write** executable code blocks (Python functions, scripts).
- These blocks can be **persisted** with metadata (name, description, execution triggers).
- They can be **invoked** by the agent or by the user, immediately or at a later point.
- They can be **composed** into multi-step pipelines or directed acyclic graphs (DAGs).

This makes the agent *reflective*: it can inspect, modify, and extend its own operational capabilities.

### 1.2 Persisted, Versioned Skill Library

When the agent writes a code block and persists it, it does not merely produce an artifact — it encodes an **operational procedure**. When it composes multiple blocks into a workflow graph, it encodes a *compound process*. The totality of the agent's saved code blocks and workflow graphs constitutes its **skill library**: a structured, executable, and growing collection of operational knowledge.

The skill library has the following technical properties:

- **Executable.** Saved skills are not descriptions — they are directly executable code. A persisted workflow is not documentation of a process; it *is* the process itself.
- **Inspectable and modifiable.** Because procedures are represented as code and graphs (not opaque neural weights), both the agent and the human can inspect, debug, and refine them.
- **Composable.** Existing skills serve as building blocks for more complex workflows. Simple skills combine into compound skills, which in turn compose into multi-stage processing pipelines.
- **Transferable.** Skills can be shared between agent instances, exported, versioned, and collaboratively developed.
- **AI-optional at runtime.** A central design principle: once a workflow graph has been constructed, it can execute without LLM involvement (**"Build with AI, Execute Deterministically"**). The agent is the *architect* but not necessarily the *executor*. The resulting workflows are standalone, deterministic automation artifacts — reproducible, auditable, and free of stochastic variance.
- **Self-improving.** The agent can revise existing procedures — replacing naive implementations with optimized ones, adding error handling, or adapting procedures to new requirements.

### 1.3 LLM-in-the-Loop Code Blocks

A critical feature is that authored code blocks are not limited to deterministic computation. A code block can invoke an LLM to perform:

- **Classification** (e.g., routing documents to processing pipelines)
- **Summarization** (e.g., condensing extracted text before further processing)
- **Translation** (e.g., language normalization in multilingual pipelines)
- **Structured extraction** (e.g., extracting tabular data from PDFs or images)
- **Decision-making** (e.g., evaluating whether intermediate output meets quality criteria)

This means code blocks are not merely scripts — they are *hybrid computational units* that blend deterministic logic with probabilistic LLM reasoning. When composed, they form **AI-augmented dataflow graphs** where some nodes are pure functions, some are LLM calls, and some are themselves small agents.

For the skill library, this is significant: the agent's skills are not limited to conventional programming. The agent can encode *judgment-requiring procedures* — tasks involving interpretation, ambiguity resolution, and context-dependent reasoning — as persisted, reusable workflow steps. A "skill" in this architecture can be something like "read an invoice PDF, extract line items into structured data, classify expenses by category, and flag anomalies" — a procedure that fundamentally requires LLM capabilities at multiple stages.

### 1.4 The State Management Problem

When composing code blocks into workflows, we face a fundamental design tension:

| Approach | Advantage | Disadvantage |
|---|---|---|
| **Universal I/O contract** (every block takes `Dict` → returns `Dict`) | Simple composition, interchangeable blocks | Loss of type safety; semantic mismatch between blocks; forces all data into flat key-value structure |
| **Shared global state** (all blocks read/write a common object) | Blocks can access any prior result | Tight coupling; namespace collisions; difficult to reason about data dependencies; no clear lifecycle for intermediate data |
| **Typed edge connections** (block outputs are typed and explicitly wired to inputs) | Clean dependencies; type safety | Rigid; hard for agents to compose dynamically; schema evolution is expensive |

The specific challenge: **ephemeral intermediate outputs**. Step *n* produces data consumed by step *n+1* but irrelevant thereafter. Storing this in a persistent shared state object pollutes the namespace, creates ambiguity about data lifecycle, and makes workflow graphs harder to analyze. Yet a purely edge-based model (typed connections) is too rigid for an agent that dynamically constructs its own workflows.

### 1.5 Why LLM-Driven Workflow Construction Is Necessary

A critical question is why an LLM is better suited here than rule-based workflow composers, genetic algorithms, or human-operated low-code tools.

The answer lies in the target audience and application context: the users of our architecture are domain experts in companies and public administration — case officers in district offices, staff in SME departments, quality managers in manufacturing — who can neither program nor invest the time to learn complex workflow design tools. These users can describe their tasks, goals, and processes in natural language, but they cannot configure DAGs, program API integrations, or define JSON schemas.

Rule-based composers require formal specifications of the desired transformation and fail in the face of the ambiguity inherent in real-world requirements. Genetic algorithms require a formal fitness function and a well-defined search space — neither of which exists for open-ended, cross-domain workflow construction tasks. Low-code tools (n8n, Zapier, Make) lower the barrier to entry but still require substantial technical understanding for configuring data flows, error handling, and API authentication — and they produce only static workflows without learning capability.

The decisive advantage of LLM-driven construction: given a sufficiently precise description of goals and tasks, the AI can develop functional solutions in extremely short time. The traditional path — a software developer laboriously coordinates with domain experts, translates their requirements into code, and iterates through feedback cycles — takes weeks to months. With our architecture, the domain expert describes the problem directly, and the agent constructs the workflow. The user's domain expertise is translated directly into automated processes without the detour through a development department.

This is not a hypothetical claim: our prototype is already actively used by customers (see Section 5), including public administrations and small-to-medium enterprises implementing digitalization projects.

---

## 2. Proposed Architecture

### 2.1 Layered State Model

We propose a **four-layer state hierarchy** inspired by scoping rules in programming languages and the blackboard architecture pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Persistent Agent Memory (incl. Skill Library)           │
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

**Layer 4 — Persistent Agent Memory.** Survives across workflow executions and sessions. This layer contains the *skill library* — the agent's authored code blocks and composed workflow graphs — alongside declarative knowledge (user preferences, domain facts) and episodic traces (execution history, success/failure records). The skill library is the core of self-extension: every successfully completed workflow can contribute a new skill to the agent's permanent repertoire.

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
    external_services: list[str]    # External APIs/systems this block connects to
    auth_requirements: list[str]    # Named credential scopes required at runtime
    triggers: list[TriggerSpec]     # Conditions under which this block should be invoked
    version: str                    # Enables skill evolution tracking
    lineage: list[str]              # Chain of blocks this was derived from
```

The `version` and `lineage` fields support skill versioning: the agent can track how skills evolve over time and maintain provenance chains when one procedure is refined into another.

### 2.3 Workflow as Graph

A composed workflow is a **directed acyclic graph (DAG)** — or, in the presence of feedback loops and conditional branching, a **directed graph with control-flow annotations**:

- **Nodes** are code blocks (with their `BlockSignature`).
- **Edges** carry typed data (Layer 2 state).
- **The workflow context** (Layer 3) is a shared annotation on the graph, accessible to all nodes.
- **Control flow** (branching, looping, error handling) is expressed through annotated edges or dedicated control-flow nodes.

The graph itself is a *first-class object in the agent's skill library*. It can be saved, named, versioned, and reused. A workflow graph is not just an execution plan — it is a **learned compound skill**.

### 2.4 "Build with AI, Execute Deterministically"

A central architectural principle: the AI agent designs and wires the workflow graph, but the resulting pipeline can execute with **zero LLM involvement** at runtime. The execution engine that traverses the graph is conventional software — not an LLM. Users, CRON jobs, or API endpoints can trigger a saved workflow directly: deterministically, without AI costs, without hallucination risk.

In practice, many workflows are hybrid: some nodes are deterministic (parsing, formatting, API calls), others invoke LLMs (classification, summarization, extraction). The `requires_llm` flag in `BlockSignature` makes this explicit per node. When LLM access is unavailable, the runtime can execute all deterministic nodes and flag or queue LLM-dependent nodes for later processing.

The economic model: the LLM is a *design-time cost*, not a *runtime cost*. A team uses the agent once to design a workflow, then runs that workflow thousands of times as a conventional automation — reproducible, auditable, and cost-neutral.

### 2.5 Skill Lifecycle

The skill library evolves through a defined lifecycle:

1. **Acquisition.** The agent encounters a novel task, decomposes it, and authors new code blocks and/or a workflow graph.
2. **Execution.** The workflow runs — triggered by the agent, manually by a user, or by an external event (CRON, webhook, API). Results are logged.
3. **Persistence.** On successful execution, code blocks and the workflow graph are saved to Layer 4 with metadata (task context, execution outcome, version).
4. **Retrieval.** When a similar task arises, the agent searches the skill library for relevant blocks and workflow patterns.
5. **Composition.** Retrieved blocks are combined — possibly with newly authored blocks — into a workflow for the new task. Existing workflows are adapted rather than rebuilt from scratch.
6. **Refinement.** Based on execution feedback, the agent modifies existing blocks (bug fixes, optimization, generalization) or restructures workflow graphs (error handling, parallelization, validation stages).

### 2.6 Secure Authentication with External Systems and APIs

Real-world workflows rarely operate in isolation. Code blocks routinely need to interact with external systems — REST APIs, databases, cloud storage, SaaS platforms, payment processors, enterprise identity providers. This makes **secure credential management** a first-class architectural concern, not an afterthought.

The challenge is amplified by several properties unique to this architecture:

- **Agent-authored code handles secrets.** The agent writes the code blocks that make authenticated API calls. This means an LLM is generating code that will, at runtime, have access to credentials. The system must ensure that the agent cannot exfiltrate, log, or inadvertently expose secrets — even though it authored the code that uses them.
- **Unattended execution.** In the "Build with AI, Execute Deterministically" paradigm, workflows run without human supervision — via CRON jobs, webhooks, or event triggers. Credentials must be available at runtime without interactive authentication flows (no browser-based OAuth redirects, no manual password entry).
- **Shared and transferred workflows.** Skills and workflows are transferable: they can be shared between agent instances, teams, or organizations. Credentials must *never* travel with the workflow. A skill exported from one environment and imported into another must bind to the *target environment's* credentials, not carry over the source's.
- **Least privilege.** A code block that reads from an API should not have write access. A block that accesses one service should not be able to reach another. The `auth_requirements` field in `BlockSignature` declares named credential scopes, enabling the runtime to enforce per-block access boundaries.
- **Credential lifecycle.** Tokens expire, API keys get rotated, OAuth grants get revoked. The runtime must handle refresh, rotation, and revocation transparently, without requiring changes to the code blocks themselves.

The current prototype implements an **authentication proxy pattern** that is already proven in production: a proxy layer intercepts outbound HTTP calls from code blocks and injects the required credentials (API keys, OAuth tokens, service account headers) at the network level. Neither the agent nor the code it authors ever sees or handles secrets directly. Even with faulty or manipulated code, credentials cannot be exfiltrated because they never exist in the code block's execution context.

Our working hypothesis is that the **proxy pattern as a base layer** is the right architectural decision, to be complemented by targeted extensions for specific deployment contexts:

- **Short-lived token minting** for time-limited access that automatically expires after a processing step completes.
- **Capability-based delegation**, where the runtime provides pre-configured, restricted client objects instead of exposing credentials directly.

Open research questions focus on: (1) the **approval chain** — who authorizes credential access when the agent itself authors the code that uses the credentials? (2) **scope escalation detection** — how to prevent the agent from rewriting code blocks to gain broader access rights? (3) the complete **audit trail** from external API call back to the agent's original design decision.

### 2.7 Data Privacy and Data Security

Data privacy and data security are central requirements of the target audience — particularly for public administrations and SMEs handling sensitive customer or employee data. The architecture addresses these requirements on multiple levels:

**Determinism as a privacy foundation.** The "Build with AI, Execute Deterministically" paradigm has a direct data-protection implication: workflows that contain no AI nodes process data in a fully deterministic and traceable manner. There are no stochastic decisions, no data transfer to external AI services, no hallucination risks. Every data processing step is reproducible and auditable — a fundamental prerequisite for GDPR compliance and sector-specific data protection requirements.

**Local AI inference.** For workflows that contain AI nodes, the interchangeability of AI models enables the use of locally hosted open-weights models (e.g., LLaMA, Mistral, Gemma) with local inference. In this mode, no data leaves the corporate network — neither during workflow design by the agent nor during workflow execution.

**Data flow transparency.** The layered state architecture makes data flows explicit: edge data (Layer 2) exists only for the duration of a step transition and is discarded afterward. Workflow context (Layer 3) is scoped to a single execution. Only data explicitly marked as persistent (Layer 4) survives across workflow executions. This clear data lifecycle management facilitates privacy impact assessments and the implementation of data deletion policies.

**Credential isolation.** The implemented proxy pattern (Section 2.6) ensures that agent-authored code never has access to credentials. This eliminates an entire class of privacy risks when integrating with external systems.

### 2.8 Agent Skills as the Packaging Standard

A central practical question is how the skill library is serialized, shared, and made discoverable. We identify a strong alignment with the **[Agent Skills](https://agentskills.io/home)** open standard ([specification](https://agentskills.io/specification)) — a lightweight, portable format for packaging agent capabilities as self-contained directory structures.

An Agent Skill is a folder containing a `SKILL.md` file (metadata + natural-language instructions), optional executable scripts, reference documentation, and static assets:

```
skill-name/
├── SKILL.md          # Metadata (name, description) + instructions
├── scripts/          # Executable code (Python, Bash, JS)
├── references/       # Documentation, technical details
└── assets/           # Templates, schemas, resources
```

The format uses **progressive disclosure** to manage context efficiently: at startup, agents load only skill names and descriptions (~100 tokens each); when a task matches, the full `SKILL.md` body is loaded; scripts and references are loaded only when actively needed during execution. This three-tier loading model directly mirrors the distinction between skill *discovery*, *activation*, and *execution* in the skill lifecycle (Section 2.5).

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

Agent Skills thus provides a **ready-made serialization format for the Layer 4 skill library**. When the agent authors a new code block and persists it, it can package it as an Agent Skill. When retrieving from the library, it follows the progressive disclosure pattern. When sharing between agent instances or teams, the format guarantees portability.

However, the current Agent Skills specification does not natively address several concerns central to our architecture:

- **Composability metadata.** There is no standard way to declare input/output schemas for a skill, making automated schema-matching between skills impossible without extension.
- **Workflow graphs.** Skills are individual units; there is no standard for packaging a *composed workflow* (a graph of skills) as a higher-order skill.
- **Edge-scoped state.** The specification does not address how data flows between skills when they are composed.
- **LLM-in-the-loop declaration.** There is no way to indicate that a skill requires LLM access or what kind of LLM reasoning it performs.
- **Authentication and credential scopes.** There is no standard way to declare that a skill requires authenticated access to external systems, what credential scopes it needs, or how credentials should be injected at runtime.

These gaps represent concrete **extension opportunities**: our research can propose additions to the Agent Skills specification that support composability, workflow packaging, and the layered state model. This positions the project not only as a consumer of the standard but as a contributor to its evolution.

Critically, Agent Skills support is **not yet implemented in [smolagents](https://huggingface.co/docs/smolagents)** — the framework our prototype is built on. Implementing this integration is a near-term deliverable that would simultaneously advance the research agenda and produce a practical, reusable open-source contribution.

---

## 3. State of the Art and Positioning

Existing open-source frameworks and research address individual aspects of the architecture described here. However, no framework integrates persistable, versioned, composable skill libraries with dynamic workflow graph construction and layered state management as a unified, production-ready system for industrial deployment. We position our work against the most relevant systems below.

### 3.1 smolagents (Hugging Face)

[smolagents](https://huggingface.co/docs/smolagents) ([GitHub](https://github.com/huggingface/smolagents)) is the first broadly adopted agent framework to implement the CodeAct pattern as a first-class paradigm: the agent writes and executes Python code as its primary mode of action, rather than merely selecting from predefined tool calls. This makes smolagents the most natural substrate for our architecture. Our prototype is built on smolagents.

However, smolagents has a critical limitation that directly motivates our work: **all agent-authored code is ephemeral**. Code is written, executed, and discarded within a single task. There is no mechanism to persist a code block for later reuse, no metadata contract for authored code, no skill library, and no workflow composition across task boundaries. Every task starts from zero. Our architecture is, concretely stated, the answer to the question: *what if smolagents code could survive beyond the task that created it?*

### 3.2 Voyager (Wang et al., 2023)

[Voyager](https://voyager.minedojo.org/) ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291), [GitHub](https://github.com/MineDojo/Voyager)) implements an LLM agent in Minecraft that writes code-based skills, persists them in a skill library, and retrieves and composes them for new tasks. This is the closest prior work to our concept of a persisted skill library.

Key differences: Voyager operates in a single domain (Minecraft) with a fixed action space; our architecture targets open-ended, multi-service workflows with heterogeneous external APIs. Voyager's skills are individual functions; we propose explicit graph-based composition with typed edges and layered state management. Voyager addresses neither authentication nor credential management nor the composability contract problem. Crucially, Voyager is a research prototype and not a production-ready open-source framework available for industrial use.

### 3.3 n8n

[n8n](https://n8n.io/) ([GitHub](https://github.com/n8n-io/n8n)) is a visual workflow automation platform with its own workflow model and execution engine. The AI Agent node is a LangChain-based tools agent that selects from connected tools within a human-designed workflow. n8n additionally offers an AI Workflow Builder and AI Transform that assist in generating workflows or code snippets. These remain, however, assistive tools for human workflow designers — the resulting workflows are static artifacts, not a self-extending skill library.

### 3.4 OpenDevin / OpenHands

[OpenHands](https://www.all-hands.dev/) ([GitHub](https://github.com/All-Hands-AI/OpenHands)) provides an open platform for coding agents with broad tool access. It does not address the specific problems of composability contracts, layered state management, or the formalization of persisted skill libraries that are central to our agenda.

### 3.5 CrewAI

[CrewAI](https://www.crewai.com/) ([GitHub](https://github.com/crewAIInc/crewAI)) provides multi-agent orchestration with task delegation and hierarchical process management. Agents execute predefined tasks via tool use; they do not generate reusable, persistable code blocks.

### 3.6 LangGraph

[LangGraph](https://langchain-ai.github.io/langgraph/) ([GitHub](https://github.com/langchain-ai/langgraph)) provides graph-based agent composition with state management and supports multi-agent workflows. Agents execute within a developer-defined graph; they do not autonomously construct or modify the graph structure itself.

| Concept | smolagents | Voyager | n8n | CrewAI | LangGraph | **Our Architecture** |
|---|---|---|---|---|---|---|
| Code authoring by agent | Yes (ephemeral) | Yes (persisted) | No | No | No | Yes (persisted + versioned) |
| Skill library | No | Yes (Minecraft) | No | No | No | Yes (cross-domain) |
| Workflow graph composition | No | No | Human-designed | Human-designed | Human-designed | Agent-constructed |
| Layered state management | No | No | No | No | Typed state channels | 4-layer model |
| Deterministic execution without LLM | N/A | No | Yes | No | No | Yes (Build with AI, Execute Deterministically) |
| Credential management | No | N/A | Yes (static) | No | No | Yes (dynamic, proxy-based) |
| Production-ready for industry | Limited | No | Yes | Limited | Limited | In active pilot deployment |

### 3.7 Summary of the Gap in the State of the Art

Existing open-source frameworks each cover partial aspects: smolagents provides CodeAct, Voyager demonstrates skill persistence in a closed domain, n8n offers workflow execution with static graphs, LangGraph provides stateful graph composition. None of these systems, however, integrates the core contributions of our architecture: (1) dynamic, agent-driven workflow graph construction from persisted, versioned code blocks, (2) a layered state architecture for composing heterogeneous processing steps, (3) secure credential management for agent-authored code with external API access, and (4) deterministic execution of the resulting workflows without LLM involvement at runtime.

Theoretical work such as Voyager demonstrates the feasibility of LLM-driven skill acquisition but is not available as a production-ready open-source framework for industrial use. This gap — between academic proof-of-concept and industry-deployable framework — is the central motivator of our work.

---

## 4. Research Agenda and Work Packages

### 4.1 Guiding Research Questions

The project focuses on four prioritized research questions:

1. **Composability contracts for agent-authored code blocks.** What level of type discipline is optimal for agent-authored code blocks? How must schema contracts be designed to enable reliable composition without over-constraining the agent's flexibility? We investigate this within the framework of **dataflow programming** as the formal foundation: code blocks as nodes in dataflow graphs, typed edges as data channels, and the layered state architecture as an extension of the classical dataflow model with scoping and context management.

2. **Skill retrieval and composition.** How should the agent index and search its skill library — by task-description similarity, by input/output schema matching, or by structural similarity of workflow graphs? Which combination yields the best reuse rates? Can the agent generalize specific procedures into reusable templates?

3. **Secure credential management for agent-authored code.** How is the approval chain for credential access designed when the agent itself authors the code? How is scope escalation through self-modification detected and prevented? What extensions of the existing proxy pattern are required for different deployment contexts?

4. **Deterministic boundary in hybrid workflows.** What fraction of real-world workflows can be fully executed without LLM involvement? Can the runtime automatically partition a hybrid workflow into deterministic sub-graphs (executable standalone) and LLM-dependent sub-graphs?

### 4.2 Work Packages

#### WP 1: Formal Foundations and Specification

We establish the theoretical framework and core abstractions:

- **Formalize the layered state architecture** — scoping rules, lifecycle semantics, and garbage collection for each layer, grounded in the dataflow programming paradigm.
  - Develop formal specification with proofs of scoping properties.
  - Elicit practical requirements from pilot customers.
  - Validate the formal model against the existing prototype.

- **Define the `BlockSignature` contract and schema language** (JSON Schema as baseline, evaluation of extensions).
  - Analyze schema expressiveness and type-theoretic properties of contract candidates.
  - Implement candidate contracts in the prototype.
  - Evaluate developer experience with each candidate.

- **Analyze the [Agent Skills specification](https://agentskills.io/specification)** and develop formal extension proposals for composability metadata, workflow graph packaging, and LLM-in-the-loop declaration.
  - Conduct gap analysis of the current specification against our requirements.
  - Design formal extensions.
  - Assess feasibility of proposed extensions against real-world constraints.
  - Optional: Contribute extension proposals to the Agent Skills open governance process.

- **Formalize the security model** — trust boundaries, scope declaration language, and approval model for credential grants.
  - Develop formal access-control model and threat-model analysis.
  - Conduct adversarial testing against the existing proxy implementation.
  - Identify deployment-specific attack vectors.

**Deliverables:** Formal specification document, extension proposals for the Agent Skills standard.

#### WP 2: Core Runtime Implementation and Integration

We build the execution engine for agent-authored workflow graphs:

- **Implement the layered state runtime** (Layer 2 + 3) with lifecycle management.
  - Build core runtime with integration into the existing prototype.
  - Verify correctness of the state lifecycle against the formal specification.

- **Implement the skill registry** with versioning and retrieval.
  - Implement registry with storage backend and API.
  - Develop retrieval algorithms (embedding-based search, schema matching, graph similarity).

- **Integrate [Agent Skills](https://agentskills.io/home) into [smolagents](https://huggingface.co/docs/smolagents)** — progressive disclosure pipeline (metadata → instructions → scripts) and agent authoring of new skills in the standard format.
  - Implement integration module and prepare open-source release.
  - Evaluate progressive disclosure efficiency and context-window impact.

- **Implement the workflow graph executor** — sequential, parallel, branching, and error-handling execution over DAGs.
  - Build executor with scheduling engine and trigger integration (CRON, webhooks, API).
  - Analyze graph-theoretic executor properties, deadlock/livelock detection.

- **Extend credential management** beyond the proxy baseline — short-lived token minting, capability-based delegation, audit logging.
  - Implement credential-management extensions with enterprise identity provider integration.
  - Evaluate security of each pattern, formally compare isolation guarantees.

**Deliverables:** Production-capable runtime with API, Agent Skills integration as open-source contribution.

#### WP 3: Agent-Driven Graph Construction and Evaluation

We enable the agent to design, modify, and optimize workflow graphs while building a growing skill library, and we evaluate the system rigorously:

- **Schema-aware block composition** — the agent reasons about `input_schema`/`output_schema` compatibility to propose valid block sequences.
  - Develop compatibility-checking algorithms and type-inference strategies for partial schemas.
  - Integrate into the agent's planning loop.
  - Design UX for human review of proposed compositions.

- **Graph construction from natural language** — decomposition of complex task descriptions into workflow graphs.
  - Implement agent-level decomposition with prompt engineering.
  - Iterate with pilot customers.
  - Systematically evaluate decomposition quality and failure modes.

- **Skill retrieval and reuse** — searching the skill library, adapting existing workflows for new tasks.
  - Develop retrieval benchmarks and reuse-rate metrics.
  - Run generalization experiments.
  - Build customer-facing skill discovery interface.

- **Industrial piloting** with 3–5 application scenarios from SME automation, manufacturing, and public administration.
  - Deploy pilots and coordinate with customers.
  - Collect structured feedback.
  - Conduct comparative benchmarking and data analysis.

- **Correctness evaluation** — benchmarks for workflow correctness, comparison against static-toolset agents and developer-authored LangGraph workflows.
  - Design benchmarks, define metrics.
  - Execute benchmark scenarios on real workloads.
  - Perform statistical analysis.

- **Security audit** — evaluation of credential management under adversarial conditions.
  - Conduct red-team evaluation and penetration testing.
  - Formally analyze scope-escalation resistance.
  - Analyze audit logs and remediate findings.

**Deliverables:** Evaluation framework, benchmark suite, industrial pilot results, analytical paper.

---

## 5. Current Prototype Status

### 5.1 Technical Foundation

The current prototype is built on **[smolagents](https://huggingface.co/docs/smolagents)** ([GitHub](https://github.com/huggingface/smolagents)), using Python as the code execution substrate. smolagents is an open-source agent framework maintained by **[Hugging Face](https://huggingface.co/)** — a company rooted in Europe (headquartered in Paris), with a large community and active open-source development. We chose smolagents for its native CodeAct support and its deliberately minimal architecture, which allows extensions without conflicts with framework conventions.

The prototype employs two complementary agent modes provided by smolagents: a **ToolUseAgent** for structured tool invocation and a **CodeAgent** for open-ended code generation and execution. The agent currently operates with four core tools: `write_code`, `read_code`, `execute_code`, and `write_metadata`. Code blocks are persisted as Python files with JSON metadata sidecars. Crucially, persistence is not limited to the code itself — metadata includes references, descriptive annotations, and **trigger definitions** that specify when a skill should execute: event-driven (e.g., incoming webhook, file upload) or time-based (e.g., CRON schedule). This means the prototype already implements a basic form of the skill lifecycle described in Section 2.5, where skills are not just stored but autonomously activated.

**Sandboxed execution.** Agent-authored Python code runs inside a multi-layered sandbox: a JavaScript-based Python interpreter (compiled to WebAssembly) executes within a JavaScript VM (Deno or Node.js). This architecture provides strong isolation — agent-generated code cannot access the host file system, network, or process space directly. All external interactions (API calls, file access) are mediated through controlled interfaces.

**File injection.** External files — images, documents, datasets — can be injected into the sandbox for use by running code or for LLM inference. This enables workflows that involve multimodal processing (e.g., image classification, document extraction) without granting the sandbox unrestricted file system access.

### 5.2 Implemented Security Model

Beyond the code-authoring tools, a complete **security model as an authentication proxy** is implemented and in production use: a proxy layer intercepts outbound HTTP calls from code blocks and injects the required credentials (API keys, OAuth tokens, service account headers) at the network level. Neither the agent nor the code it authors ever sees or handles secrets directly. This demonstrates that the security aspect of the architecture is not an exploratory research goal but is already concretely implemented and proven in deployment.

### 5.3 Active Development: Shared State Between Code Blocks

The next immediate engineering effort — currently in implementation — targets **shared state between multiple executables**. Today, each code block runs in isolation; passing data between blocks requires explicit serialization at the caller level. We are evaluating state-passing patterns inspired by LangGraph (explicit state objects threaded through graph edges) and LangChain (shared memory abstractions) to determine the best fit for our layered state model. This work directly addresses Layers 2 and 3 of the proposed architecture (Section 2.1): edge-scoped data flow for sequential handoffs and workflow-scoped context for cross-cutting shared data. The outcome of this implementation will serve as the empirical foundation for the formal specification work planned in WP 1.

### 5.4 Maturity and Active Use

The framework has reached a maturity level that enables active customer deployment. Current users include:

- **Public administration:** District offices (Landratsämter) use the prototype for digitalization initiatives.
- **Small and medium-sized enterprises (SMEs):** Automation of data processing and document workflows.
- **Select larger enterprises:** Pilot projects for more complex automation scenarios.

The primary customer base consists of SMEs and public institutions pursuing digitalization projects. This active use demonstrates the fundamental feasibility and practical value of the approach. The full potential of the architecture — particularly skill composition, layered state management, and formal foundations — will only be realized through the further developments planned in this research project.

### 5.5 Target Audience and Application Scenarios

The architecture addresses three primary application areas with high relevance for the Bavarian economy:

**SME automation:** SMEs that require complex data-processing workflows but lack development teams. The agent designs the automation, which then runs deterministically without AI costs. Domain experts in administration, procurement, or quality assurance describe their processes in natural language; the agent translates these into executable workflows.

**Manufacturing and Industry 4.0:** Workflow creation for quality control, supply chain monitoring, and predictive maintenance pipelines. The combination of deterministic processing steps and optional AI nodes (e.g., image classification for quality inspection) fits precisely the requirements of manufacturing industry.

**Administration and bureaucracy:** Automation of document processing, application handling, and multilingual communication. Public administrations benefit particularly from the deterministic execution paradigm: all data processing is traceable and reproducible when no AI is used in the workflows.

### 5.6 Preliminary Results: End-to-End Complaint Management Example

As a concrete application example, we present a complaint management workflow that demonstrates the full lifecycle of the architecture:

1. **Input:** A customer photographs a defective product and submits the photo through a defined channel.
2. **Multimodal extraction:** A vision AI model extracts relevant information from the photo — serial number, type of defect, affected component.
3. **Defect assessment:** A multimodal AI model evaluates the defect based on visual features (severity, safety relevance).
4. **Classification and decision:** An LLM makes a classification — replacement or repair — based on the defect assessment, warranty status, and company policies, following defined instructions and criteria.
5. **Human-in-the-loop:** The AI decision is presented to an employee for review and approval.
6. **Process initiation:** Upon approval, the repair or replacement process is automatically triggered.
7. **Real-time feedback:** The status is communicated back to the customer in real time.

This workflow demonstrates several core architectural principles: hybrid processing (deterministic steps + AI nodes), human-in-the-loop integration, the layered state architecture (image data at the edge level, case context at the workflow level, criteria rules in the persistent skill library), and reusability — the complaint management workflow, once constructed, runs deterministically for every new case.

## 6. Interchangeability and Technology Independence

### 6.1 Agent Framework

The architecture is deliberately framework-agnostic. Its core contributions — the layered state model, the `BlockSignature` contract, the workflow graph executor, the skill lifecycle, the credential proxy, and the Agent Skills integration — sit *above* the agent framework layer. smolagents can be replaced by any other framework (LangChain, CrewAI, AutoGen, or a custom implementation) without redesigning the architecture.

### 6.2 AI Models

The AI models used are freely interchangeable — as a commodity: from commercial providers (OpenAI, Anthropic, Google) to European providers (Mistral AI, Aleph Alpha) to locally hosted open-weights models (LLaMA, Mistral, Gemma) with local inference for maximum data security and privacy. The architecture makes no assumptions about the model used; the agent framework layer fully abstracts the model choice.

For privacy-sensitive applications — particularly in public administration and companies with strict compliance requirements — local inference with open-weights models ensures that no data leaves the corporate network.

### 6.3 Agent Skills as an Open Industry Standard

The **[Agent Skills](https://agentskills.io/home)** standard ([specification](https://agentskills.io/specification)) is developed as an open industry standard with open governance. It is published as open source and maintained by an open community. This guarantees vendor independence and long-term availability of the serialization format for the skill library. Integrating the standard into smolagents is a near-term goal that simultaneously advances the research agenda and produces a practical open-source contribution to the broader agent ecosystem.

---

## 7. Economic Benefit for the Bavarian Economy

### 7.1 Strengthening Digitalization in the Bavarian Mittelstand

Bavaria is characterized by a strong Mittelstand with high digitalization demand. Many SMEs possess deep domain expertise but lack the IT resources to develop complex automation solutions. Our architecture addresses precisely this gap: domain experts describe their processes in natural language, the agent constructs the automation, and the resulting workflows run deterministically without AI costs. The result is a drastic reduction of the barrier to entry for process automation.

### 7.2 Concrete Value Creation

- **Cost reduction for SMEs:** Automations that previously required weeks of developer time can be created in hours. The resulting workflows run without ongoing AI costs — the LLM is a one-time design expense.
- **Workforce relief:** In times of skilled-labor shortages, the architecture enables existing employees to work more efficiently rather than requiring additional IT specialists.
- **Administrative modernization:** Public administrations in Bavaria can automate application and document processing workflows — traceably, deterministically, and in compliance with data protection regulations.
- **Industry 4.0 workflows:** Bavarian manufacturing companies gain a tool for rapid creation of quality control and monitoring pipelines.

### 7.3 Open-Source Contributions and Standardization

The open-source contributions produced by the project (features and improvements in smolagents or other agent frameworks, Agent Skills integration in smolagents, extension proposals for the Agent Skills standard) position Bavarian research — anchored at the University of Bayreuth — and industry as active shapers of the European AI ecosystem — rather than passive consumers of American or Chinese technology platforms.

---

## 8. Technology Readiness Levels and Milestones

### 8.1 TRL Classification

| Component | Current TRL | Target TRL (End of Project) |
|---|---|---|
| Code authoring and persistence | TRL 7 (in deployment) | TRL 8 |
| Authentication proxy | TRL 7 (in deployment) | TRL 8 |
| Sandboxed execution (WASM-based) | TRL 7 (in deployment) | TRL 8 |
| Trigger system (event/CRON) | TRL 7 (in deployment) | TRL 8 |
| File injection into sandbox | TRL 7 (in deployment) | TRL 8 |
| Shared state between code blocks | TRL 3 (in implementation) | TRL 6 (pilot) |
| Layered state architecture (formal) | TRL 2 (conceptualized) | TRL 5 (validated) |
| Workflow graph executor | TRL 3 (proof of concept) | TRL 6 (pilot) |
| Agent Skills integration | TRL 1 (basic research) | TRL 5 (validated) |
| Skill retrieval and composition | TRL 2 (conceptualized) | TRL 5 (validated) |

### 8.2 Milestones

| Milestone | Verifiable Deliverable |
|---|---|
| Formal specification completed | Specification document, positioning paper? |
| Layered state runtime implemented | Working prototype with Layer 2+3, unit tests |
| Agent Skills integration in smolagents | Open-source release, functional proof with 5 skills |
| Workflow graph executor with piloting | 3 industrial pilot scenarios completed |
| Evaluation framework and benchmarks | Benchmark suite, comparative evaluation published |
| Project completion | Final paper, open-source releases, exploitation plan |

### 8.3 Exploitation Plan

The core technology produced by this project will be released as **open source**. The commercial model is built on top of — not instead of — open contributions:

- **Short-term (during project duration):** Integration of research results into the existing prototype; expansion of the customer base through improved workflow composition. Continued open-source contributions to smolagents and the Agent Skills standard.
- **Medium-term (1–2 years post-project):** Commercial **Platform-as-a-Service (PaaS)** offering that wraps the open-source runtime with a user-friendly frontend, managed infrastructure, professional support, consulting, and integration services. The value proposition is not the technology itself — which remains open — but usability, reliability, and domain-specific expertise.
- **Long-term:** Positioning as a driving force behind an evolving agent ecosystem and open standards for increasingly powerful agent architectures. Establishment of a marketplace for composable and tradeable skills built on the Agent Skills standard.

### 8.4 Demonstrated Impact and Industry Validation

The project team has already demonstrated the ability to shape the broader agent ecosystem through open-source contributions. In 2025, we designed and implemented **output schema and structured output support for smolagents**, which was [merged by Hugging Face into the main framework](https://huggingface.co/blog/llchahn/ai-agents-output-schema). This contribution directly improved the reliability of agent-produced outputs — a prerequisite for the deterministic execution paradigm central to our architecture.

More broadly, the CodeAct approach that our architecture builds upon has since been adopted by major industry players, validating the technical direction: **Cloudflare** integrated code-based agent execution into their [TypeScript agent framework](https://blog.cloudflare.com/code-mode/), and **Anthropic** adopted programmatic tool calling in [Claude's agent platform](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) as well as [code execution via MCP](https://www.anthropic.com/engineering/code-execution-with-mcp). These independent adoptions confirm that the code-authoring agent paradigm is not a niche research direction but an emerging industry standard — and that our early investment in this approach positions the project at the leading edge of the field.

---

## 9. Risk Assessment and Mitigation

### 9.1 Technical Risks

| Risk | Assessment | Mitigation |
|---|---|---|
| smolagents introduces incompatible API changes | Low | Modular design: smolagents is a replaceable component. The core architecture sits above the framework layer and can operate with any available open-source agent framework — not only CodeAct-based ones. As a further fallback, the MIT-licensed smolagents codebase can be forked and maintained as an independent branch if mainstream development diverges from the project's requirements. |
| Agent Skills standard evolves incompatibly | Low | Active co-design through open governance. The project consortium is a contributor, not merely a consumer of the standard. |
| LLM quality insufficient for workflow construction | Medium | The architecture is model-agnostic; AI models are treated as a commodity. Improvements in foundation models automatically benefit the architecture. |
| Security vulnerabilities in agent-authored code | Medium | The implemented proxy fully isolates credentials. Runtime sandboxing and scope enforcement will be further developed in the project. |

### 9.2 Dependency Risks

The project has deliberately minimal external dependencies:

- **smolagents:** Open source (MIT license), maintained by Hugging Face — an international but distinctly European company, rooted in Europe and headquartered in Paris. Large community, active development. In the worst case, the codebase can be forked under its MIT license, or replaced entirely with an alternative open-source agent framework.
- **Agent Skills:** Open standard with open governance. No dependency on a single company. The project consortium (University of Bayreuth and industry partner) actively participates in standard development.
- **AI models:** Fully interchangeable. From commercial APIs to European providers to local open-weights models — the project is not tied to any model provider.

### 9.3 Economic Risks

Risk is low due to the modular architecture and proven market resonance. The prototype is already in active customer use; the research project extends a functioning system rather than building a new one from the ground up.
