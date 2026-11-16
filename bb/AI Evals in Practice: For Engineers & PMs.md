# AI Evals in Practice: For Engineers & PMs
<img width="384" height="353" alt="image" src="https://github.com/user-attachments/assets/660c5074-4e64-4e2e-85b3-f13fc09cbc25" />

## What you'll learn

### Understand how AI evals work
* Learn why unit tests and traditional QA break on non-deterministic systems
* Measure AI quality like accuracy, safety, reliability, cost, and latency
* Turn real failures into your first evals with error analysis
* Take a multi-step agent trace and find where it went wrong in minutes, not days.

### Build LLM judges you trust
* Grade outputs automatically with LLM-as-a-Judge
* Write pass/fail rubrics that behave like assertions
* Validate every judge against human review before relying on it

### Red-team agents before attackers do
* Attack your own agent with prompt injection and jailbreaks
* Automate red teaming with Promptfoo
* Add guardrails that block attacks without blocking real users
* Put your agent through a safety and adversarial review your security team takes seriously.

### Build meaningful eval datasets
* Build golden datasets from real traffic plus validated synthetic data
* Version datasets like code so results stay reproducible
* Score RAG pipelines with the RAGAS metrics

### Evaluate agents on different dimensions
* Test whether agents call the right tools with the right arguments
* Judge the agent's full trajectory and final answer
* Catch failures hiding in multi-agent handoffs

### Run evals in production
* Wire evals into CI/CD so regressions never merge
* Monitor live traffic
* Catch model and data drift

### Ship a complete eval system
* Take a capstone from untested agent to gated, monitored pipeline
* Pick the cheapest model that still meets your quality bar
* Turn every incident into a permanent regression test

## Eval Foundations and Error Analysis
### Why AI Products Need Evals
* Production failure case studies
* Model benchmarks vs. product evals
* Non-determinism and how to approach it
* Traditional QA vs. AI evals
* Evals as core infrastructure
### The Fundamentals of AI Evaluation
* The six quality dimensions: model quality, product behavior, safety, reliability, cost, latency
* Model quality vs. product behavior evals
* Cost metrics: per-task cost tracking, token and retry accounting
* Latency metrics: time-to-first-token, span timing, percentile tracking
* Eval harness architecture and pipeline components
* Eval framework landscape (open-source and commercial)
### Build a Real Agent
* Create a real customer service agent so every run leaves a trace you can inspect.
* Set up logging and observability that shows what the agent actually did.
### Error Analysis
* Traces vs. logs
* The error analysis workflow
* Failure taxonomies
* Frequency-based prioritization
* The eval flywheel: deploy → analyze → build dataset → improve → monitor
### Assignment 1: Instrument and trace
* Instrument the code assistant with the LangSmith SDK
* Run 30 inputs: code-gen, doc lookup, multi-turn
* Annotate traces end-to-end with pass/fail and notes
### Assignment 2: Build a failure taxonomy
* Cluster failures into 5+ binary trace codes with frequency counts
* Map each code to a quality dimension
* Report the top 3 failure modes and 1 unexplained anomaly
* Eval Foundations and Error Analysis diagram
<img width="500" height="489" alt="image" src="https://github.com/user-attachments/assets/cc756f1f-ecb8-4a45-afec-a97a27425f08" />

## Designing Reliable LLM Judges
### Evaluation Methods
* Evaluator types (code-based, traditional metrics, LLM-as-a-Judge, human eval)
* Failure modes of traditional metrics
* Evaluator selection: accuracy, cost, and latency trade-offs
### Designing an LLM Judge
* Binary pass/fail rubrics vs. Likert scales
* Judge prompt anatomy
* Single-criterion judges for root-cause isolation
* The four judge types: tone/style, factual correctness, instruction following, task completion
### Validation and Comparison
* Judge–human agreement matrices
* False-pass vs. false-fail analysis
* Trust gates, false-pass rate, explainable failures
* Multi-turn and conversation-level evaluation
* Per-metric A/B evaluation
### Assignment 1: Build and validate
* Write 3 binary rubrics: code correctness, doc relevance, tool-call accuracy
* Implement as LLM judges in DeepEval with strict output formats
* Validate on the 40-label set and look into agreement and Cohen's κ
### Assignment 2: Diagnose and compare
* Run a diagnose-and-fix cycle
* A/B test two prompt variants with per-metric deltas
* Write a ship/no-ship decision
* Designing Reliable LLM Judges diagram
<img width="457" height="372" alt="image" src="https://github.com/user-attachments/assets/94b37c14-743d-457b-85e1-b81e68f9fd22" />

## Safety and Adversarial Testing
### The Adversarial Foundations
* Adversarial evaluation vs. quality evaluation
* The agent attack surface
* Attack categories like prompt injection, jailbreaks, data leakage, harmful content, and bias
### Attack Techniques
* Direct vs. indirect prompt injection
* Jailbreak techniques: role-play, hypotheticals, emotional framing, encoding
* Data leakage targets
* Bias testing: demographic prompt variation
* Harmful-content testing
### Building the Suite and Scoring Guardrails
* Manual vs. automated red teaming
* Automated red teaming with Promptfoo
* Test suite schema: attack type, target, expected behavior
* Guardrail confusion matrices
* Guardrail composition
### Assignment 1: Scan and triage
* Run a Promptfoo baseline scan
* Triage failing traces into a frequency × severity matrix
* Craft 5 domain-specific attacks by hand
### Assignment 2: Build and score guardrails
* Build 3 guardrails (regex, LLM judge, or hybrid)
* Score each on catch rate and over-block rate
* Write a ship/no-ship verdict with numeric thresholds
* Human-in-the-loop approval flows
* Safety and Adversarial Testing diagram
<img width="464" height="478" alt="image" src="https://github.com/user-attachments/assets/535e1641-914a-484a-9be4-14db6387b95b" />

## RAG Evaluation
### Eval Data
* The four dataset qualities: representativeness, diversity, label correctness, versioning
* Golden-set schema
* Criteria-based ground truth
* Synthetic data generation and validation
* Dataset versioning and pinning in LangSmith
* Benchmark literacy (MMLU, HumanEval, SWE-bench)
### RAG Evaluation
* Component-wise evaluation: retrieval vs. generation
* The four RAGAS metrics: faithfulness, answer relevance, context precision, context recall
* Groundedness
* RAGAS + LangSmith integration
### Statistical Rigor
* Sample size and margin of error
* Paired testing
* Bootstrap confidence intervals
* Significance testing and p-values
* Multiple comparisons and held-out re-testing
* Repeatable harness design (pinned datasets, one config, reproducible runs, etc.)
### Assignment 1: Build and version the dataset
* Build a 150+ case golden set
* Pin v1 in LangSmith for versioned A/B runs
### Assignment 2: Measure and test
* Wire the four RAGAS metrics into the harness
* Find one groundedness failure
* Run a paired bootstrap test
* Ship the harness with one config, one command, and a reproducible report
* RAG Evaluation diagram
<img width="462" height="608" alt="image" src="https://github.com/user-attachments/assets/2331f1d2-df01-4438-a6f1-f4e3792b2fe7" />

## Multi-Agent Evaluation
### Agent Eval Foundations
* model, agent, and system
* Agent failure modes: tool misuse, reasoning loops, memory drift, permission escalation, coordination conflict
* Turn-level vs. conversation-level metrics
### Tool Use and Trajectories
* Tool-call scoring
* Expected-action sets
* Argument validation as structured-output evaluation
* Trajectory evaluation
* Property-based trajectory judging vs. exact-match diffs
### Multi-Agent Coordination
* Handoff evaluation (completeness, fidelity, routing)
* Cascading failures in agent pipelines
* Model→system score gaps as interaction-failure signals
### Assignment 1: Score tools and trajectories
* Build 15+ expected-action records with order and extra-call policies
* Score tool use per dimension
* Build a trajectory evaluator
* Flag one fragile pass
### Assignment 2: Find interaction failures
* Wire the code-gen review handoff
* Produce a three-layer report for model, agent, and system
* Document 3 interaction failures with handoff root causes
* Multi-Agent Evaluation diagram
<img width="480" height="592" alt="image" src="https://github.com/user-attachments/assets/4268d68a-0df2-45eb-aa2f-b8e03a64a1c9" />

## Production Eval Infrastructure
### Eval Gates in CI/CD
* Eval checkpoints (PR check, nightly regression, pre-launch gate)
* Eval gates in GitHub Actions
* Gating policy: blocking vs. logged metrics
* Fast PR subsets vs. full regression suites
* Online Evaluation and Drift Detection
### Online evaluation signals
* Traffic sampling for eval cost control
* Alerting policy
* The drift taxonomy (data drift, concept drift, prompt/model drift)
* Scheduled canary evaluation
### Compounding Eval
* The eval flywheel: from production failures to regression tests
* Eval-driven model routing and cost optimization
* Ownership and operating cadence
### Assignment 1: Deploy the infrastructure
* Integrate the Weeks 1–5 harness into GitHub Actions: PR check + nightly cron, gated on safety and correctness
* Deploy production monitoring
* Schedule a drift canary with alerting
### Assignment 2: Operate and document
* Document one flywheel cycle like incident → root cause → fix → regression test
* Produce an eval-driven cost-routing recommendation
* Write the ownership doc
* Production Eval Infrastructure diagram
<img width="494" height="482" alt="image" src="https://github.com/user-attachments/assets/015a8557-00d6-4b26-a8f3-ec198b7548d7" />

* https://live.bytebytego.com/courses/ai-evals?promotionCode=INSIDER20
