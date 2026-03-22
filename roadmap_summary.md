# Project Proposal: Dynamic Tool Synthesis and Graph-Based State Management in Self-Extending Agentic Workflows

## 1. Abstract
Recent advancements in Large Language Models (LLMs) have driven the development of autonomous agents capable of utilizing external tools (ReAct) and writing executable code (CodeAct). However, these paradigms often treat code generation as a transient action or a final artifact. We propose a hybrid architecture where an agent dynamically synthesizes, persists, and orchestrates code blocks as reusable, composable tools using the open **Agent Skills** specification (`agentskills.io`).

As the agent solves novel problems, it compiles its solutions into standardized Skill packages and chains them into Directed Acyclic Graphs (DAGs). Crucially, this architecture supports **Decoupled Execution**: once a graph is synthesized, it can be orchestrated autonomously by the AI *or* triggered manually as a standalone, deterministic pipeline by a human user or external system without any LLM involvement. 

A critical unresolved challenge in this self-extending architecture is data routing and state management between these dynamically generated nodes. This research project aims to investigate and formalize state management paradigms for dynamic workflows, culminating in a framework that seamlessly handles both global context and transient, step-to-step data passing, regardless of whether the graph is executed by an AI agent or a standard deterministic runtime.

## 2. Core Architecture & Innovations
Our foundation is built upon a modified `smolagents` architecture, specifically extending a `ToolUseAgent` with the following capabilities:

* **Standardized Persistent Tool Synthesis (Agent Skills):** The agent continuously expands its action space by compiling useful generated code into standard Agent Skill directories (`SKILL.md` instructions, executable `/scripts`, and context `/references`).
* **Hybrid ReAct & CodeAct:** The agent seamlessly transitions between utilizing pre-existing tools, writing ad-hoc code for immediate execution, and packaging successful routines into permanent Agent Skills.
* **Decoupled Execution Modality ("Build with AI, Execute Deterministically"):** The synthesized workflows are not inextricably linked to the LLM orchestrator. If a composed graph consists entirely of deterministic code blocks (e.g., standard data processing or PDF generation), it can be manually triggered, completely bypassing AI compute overhead and hallucination risks.
* **Recursive Agentic Tools (Sub-Agents):** For non-deterministic tasks, synthesized skills can optionally wrap specific LLM calls. This creates a nested architecture where a manually triggered graph might contain localized "pockets" of AI, or an AI orchestrator might manage a graph of purely deterministic tools.
* **Dynamic Graph Construction:** For complex tasks, the agent acts as a compiler, chaining its repository of Agent Skills into cohesive, reusable pipelines.

## 3. The Core Research Challenge: State Management in Dynamic Graphs
As the agent constructs multi-step workflows linking multiple generated Agent Skills (e.g., Skill A -> Skill B), standardizing the Input/Output (I/O) becomes the primary bottleneck. This challenge is compounded by the **Decoupled Execution** requirement: the state management system must function identically regardless of whether an LLM or a standard Python runtime is executing the graph.

Currently, two dominant paradigms exist, both with severe limitations for dynamic synthesis:

* **Universal Global State (The Blackboard Pattern):** All skills read from and write to a single, shared state object. 
    * *Drawbacks:* Inefficient for transient, intermediate data. It bloats the LLM context window (if the AI is orchestrating) and risks data overwriting or state corruption in complex graphs.
* **Strict Universal I/O Interfaces:** All skills must accept and return a highly rigid data envelope.
    * *Drawbacks:* Highly restrictive for an agent writing code dynamically. It forces the LLM to write excessive boilerplate for simple data transformations during the synthesis phase.

**The Research Question:** *How can we design a hybrid state-management topology that allows dynamically generated Agent Skills in a graph to securely pass strongly-typed transient data to adjacent skills via edges, while maintaining a lean state model that functions seamlessly in both agent-driven and purely deterministic (manual) execution environments?*

## 4. Proposed Research Roadmap

### Phase 1: Foundations & The Agent Skills Adapter
* **Objective:** Integrate the Agent Skills specification into the `smolagents` framework and formalize the graph model for skill chaining.
* **Tasks:**
    * Build an adapter for `smolagents` that natively loads and executes standard Agent Skills via the `SKILL.md` format.
    * Develop a **Runtime-Agnostic State Protocol**: A hybrid state model encompassing Global Context (read-only for all nodes) and Edge State (transient data passed explicitly) that works independently of the execution trigger.

### Phase 2: Dynamic Skill Synthesis and Orchestration
* **Objective:** Enable the orchestrator agent to autonomously write new Agent Skills and construct verifiable data-flow graphs.
* **Tasks:**
    * Develop the mechanism for the agent to scaffold new skills, writing the YAML frontmatter, markdown instructions, and underlying scripts.
    * Implement runtime I/O validation using the Agent Skill `/references` specification, allowing the graph's data routing to be validated *before* manual or agentic execution.

### Phase 3: Decoupled and Recursive Execution Integration
* **Objective:** Stabilize the hybrid nature of the nodes (deterministic code vs. sub-agents).
* **Tasks:**
    * Implement the core architecture allowing a user or external API to manually trigger a previously AI-generated graph.
    * Study the emergent behavior of orchestrator agents managing sub-agents, alongside performance profiling of manually triggering those same AI-embedded workflows.

### Phase 4: Evaluation and Framework Generalization
* **Objective:** Benchmark the architecture against static setups and prepare publications.
* **Tasks:**
    * Evaluate the framework's efficiency: Compare the computational overhead (time, token cost) of an agent solving a task from scratch versus a human manually executing the AI's previously synthesized graph for the same task.
    * Finalize a joint research paper detailing the graph-based state management solution for self-extending, execution-agnostic workflows.
