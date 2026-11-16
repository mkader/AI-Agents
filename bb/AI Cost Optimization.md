# AI Cost Optimization
<img width="374" height="314" alt="image" src="https://github.com/user-attachments/assets/421c3165-2c81-489d-bb95-ab19b25d9459" />

## What you'll learn
### Understand your AI bill
* Know what each token type costs and why
* See how a short question turns into thousands of billed tokens
* Do the napkin math on what a request costs

### Track spend across your org
* Put one gateway in front of all your AI calls
* Break spend down by feature, team, and user
* Find the one or two features driving most of the bill

### Cut costs with caching
* Arrange prompts so provider caches actually hit
* Catch the small mistakes that quietly break caching
* Answer repeat questions without calling the model

### Keep agent context small
* Learn why stale tool output is the first thing to cut
* Summarize long histories without dropping key facts
* Store agent memory outside the context window

### Replace big prompts with small models
* Understand how a small model can take over one high-volume job
* Stop paying for the same long prompt on every call
* Learn what breaks when the small model goes stale

### Route requests to the right model
* Send easy requests to cheap models and save the big ones
* Build a router and its evals from scratch
* Compare routers on cost and quality, not demos

### Stop agents from burning money
* Estimate multi-agent costs before you build
* Catch loops early, before the retries pile up
* Fix a broken research agent as your capstone

## AI Cost Dynamics and Observability
### AI cost foundations
* Core meters (input, output, cached, batch, and reasoning tokens)
* Secondary meters (tool use, images/audio/video, embeddings)
* Purchasing models (pay-as-you-go, provisioned throughput, self-hosted)
* Economics of LLM inference
* Latency metrics like TTFT and inter-token latency
* Tokenomics in agentic systems
* Cost amplifiers: tool schemas, RAG documents, multi-agent fan-out, etc.
### Cost observability
* Context growth and failure modes like context rot
* Traces and spans for LLM applications
* OpenTelemetry GenAI semantic conventions
* Content vs. metadata logging (privacy, PII, retention)
* Sampling strategies
* Attribution hierarchy
* Unit economics such as cost per conversation or task
### Gateways and where to optimize
* LLM gateway / LLM mesh like routing and budgets
* Gateway landscape (LiteLLM, Portkey, Kong, Cloudflare, OpenRouter) and trade-offs
* Observability stacks (Langfuse, LangSmith, Helicone, Arize Phoenix, W&B Weave, Braintrust)
* Build vs. buy
* Baseline diagnostics (top drivers, cache hit rate, waste rate)
* When not to optimize
* Optimization stack overview, ranked by ROI
### Assignments: Instrument an LLM Gateway
1. Build the mesh
* Build a gateway for a multi-tenant workload
* Emit telemetry for every call (tokens, latency, attribution)
* Connect an observability stack (Langfuse)
2. Diagnose and optimize
* Produce a cost-diagnosis report
* Ship one optimization with a measured before/after
* AI Cost Dynamics and Observability diagram - Week 1 assignment diagram
* <img width="454" height="544" alt="image" src="https://github.com/user-attachments/assets/4e7f7f1f-acbb-4bb4-a3aa-5b27d57eabe2" />

## Token Efficiency Techniques
### Caching
* Exact-match response caching
* Semantic caching (embed → ANN search → threshold → serve or miss)
* Threshold tuning for cache hits
* Failure taxonomy
* Hardening like scoped keys and verifier models
* Provider prompt caching
* Stable-to-volatile ordering of input prompt
* Cache-busting anti-patterns
### Context compression
* Compaction vs. caching
* Agentic compaction (/compact-style)
* Tool-result pruning
* Attention sinks and window management
* Hard compression and soft compression
* KV-cache eviction and quantization
* Context OS
### Distillation
* Prompt and context distillation
* Model distillation (soft labels, sequence-level KD, SFT on teacher outputs)
* Limitations and losses
### Assignments: Practice Token Efficiency
1. Semantic cache
* Build a semantic cache and sweep similarity thresholds
* Harden with a policy gate and a verifier model
* Build a prefix-cache simulator and detect cache-busting anti-patterns
2. Compaction
* Compare compaction strategies with a needle-recall test
* Measure cost vs. recall per strategy
3. Distillation
* Distill a big-prompt router into a lightweight student
* Test staleness after a policy change and retrain
* Token Efficiency Techniques diagramWeek 2 assignment diagram
<img width="457" height="505" alt="image" src="https://github.com/user-attachments/assets/6210849f-1485-42cb-b383-a1089311f039" />

## Intelligent Routing and Model Cascades
### Model and reasoning effort economics
#### Reasoning effort as a cost lever
* Capability overlap and difficulty prediction
* Model complementarity (RouterBench, RouterArena)
* Cost–quality–latency trade-offs
#### Model routing
* Model cascades (FrugalGPT)
* Expected-cost formulation
* Routing frameworks (WHEN × SIGNAL × COMPUTE)
* Predictive vs. non-predictive routing, hybrid systems
* Ecosystem (RouteLLM, LiteLLM, OpenRouter, Azure Model Router, Bedrock, etc)
* RouteLLM deep dive (router architectures, data augmentation, calibration, serving)
* Deterministic and vector-based routing
* Agent-step routing
#### Routing evals
* Metrics (PGR, APGR)
* Offline certification (golden sets, replay matrix)
* Programmatic checks and LLM-as-judge labeling
* Online monitoring (routing distribution, shadow scoring, canary, interleaving)
#### Assignment: Build a Router
* Build a model router from scratch
* Run it against a simulated model pool
* Build evals and certify the router
* Intelligent Routing and Model Cascades diagramWeek 3 assignment diagram
<img width="450" height="658" alt="image" src="https://github.com/user-attachments/assets/e1452cc1-41aa-422f-9411-562e44d522ef" />

## Agentic Workflow Optimization
### Multi-agent loops
* Decomposition (context isolation, parallelism, specialization)
* Loop topologies (orchestrator, evaluator, peer debate, blackboard)
* Cost models for multi-agent systems
### Runaway loops and state hygiene
* Runaway taxonomy (retry spirals, ping-pong delegation, oscillation, goal drift, amplification)
* Detection (counters, state hashing, semantic repetition, progress metrics, watchdog agents)
* Context rot (attention dilution, behavioral contamination)
* State cleansing (truncation, retention policies, compaction, external memory, fresh restarts)
* Error hygiene (digests, deduplication, failures → constraints)
### Planning, speed, and systems thinking
* Semantic convergence (embedding similarity, edit distance, evaluator score)
* Convergence vs. stagnation
* Planning (ReAct vs. plan-then-execute, tree search, pruning, replanning)
* Speculative decoding (draft–verify, acceptance math, agent-level speculation)
* Co-design (tool, model, prompt, eval, infra)

* https://live.bytebytego.com/courses/ai-cost?promotionCode=INSIDER20
### Capstone: Rescue a Research Agent
* Redesign a failing deep research agent
* Apply the full optimization toolkit end to end
* Agentic Workflow Optimization diagramWeek 4 assignment diagram
<img width="444" height="501" alt="image" src="https://github.com/user-attachments/assets/ed49fb79-34c6-49f1-9395-fc1516121889" />

* https://live.bytebytego.com/courses/ai-cost?promotionCode=INSIDER20
