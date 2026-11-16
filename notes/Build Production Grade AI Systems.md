# Build Production Grade AI Systems
<img width="360" height="293" alt="image" src="https://github.com/user-attachments/assets/9d6ab9f8-1246-48e4-838b-63291ead861f" />

## What you'll learn

### Ship prototypes as production services
* Turn a notebook script into a layered FastAPI service
* Package versioned model artifacts into Docker images
* Add retries, health checks, and structured logging

### Build enterprise RAG pipelines
* Chunk, embed, and index real company documents
* Combine keyword and semantic search so the right document always surfaces
* Catch bad data before it reaches your users

### Build reproducible model pipelines
* Track experiments so any result can be rebuilt and trusted
* Know when to prompt, when to retrieve, and when to fine-tune
* Get better models without burning your compute budget

### Cut inference latency and cost
* Make responses faster with caching and batching
* Understand where LLM serving time goes
* Decide between paid APIs and your own GPUs with real cost numbers

### Evaluate and monitor production AI
* Gate releases on automated quality checks
* Catch agents that forget context or call the wrong tool
* Spot quality slipping in production before users complain

### Secure and govern AI systems
* Ensure the assistant only answers from documents each user may see
* Block prompt injection and jailbreak attempts with layered defenses
* Ship prompt changes with review, testing, and rollback

### Operate AI platforms at scale
* Release to a small slice of traffic first and roll back automatically
* Add human approval before agents take high-risk actions
* Present a complete system you can explain, cost out, and defend

### From Prototype to Production
#### Production Architecture and the AI Lifecycle
* Apply production patterns to two tracks throughout the course: traditional ML systems (fraud detection, recommenders) and LLM applications (RAG, agents).
* Prototype vs production systems; training/serving skew
* AI lifecycle from data to packaging, serving and monitoring
* Handoff artifacts
* Batch vs online inference, monolith vs microservices vs event-driven
* Production building blocks (API gateways, feature stores, vector DBs, model registries)
* Reliability primitives like retries, timeouts, queues, and graceful degradation
* REST vs gRPC; streaming responses
#### Production Services, Packaging, and Containers
* Layered service architecture (API / domain / model adapters)
* Configuration management, secrets separation, request IDs, retry budgets
* Async programming for I/O-bound LLM and retrieval calls
* Structured logging; health and readiness endpoints
* Model serialization: versioned artifacts with sha256 metadata
* Packaging path: notebook → artifact → API → container image
* Docker fundamentals; Kubernetes overview
#### Live Demo: Turn a Python script into a versioned, containerized AI service
<img width="518" height="662" alt="image" src="https://github.com/user-attachments/assets/1b1229f4-2fe9-486e-b8ff-886cfdf2ba6f" />

### Data and Model Pipelines
#### Data Pipelines and Enterprise Retrieval
* Ingestion patterns (batch vs streaming, ETL vs ELT)
* Data validation gates and schema enforcement
* Data lineage and versioning
* Training/serving parity; leakage detection
* Deep-learning data paths (loaders, batching, augmentation, distributed training)
* Document parsing (OCR, PDF, HTML) and chunking strategies
* Vector DB operations (indexing, upserts, metadata filters)
* Context packaging (ordering, token caps, citations)
* Hybrid retrieval: BM25 + dense embeddings + reranking; GraphRAG
#### Model Development and Reproducibility
* Experiment tracking (MLflow, Weights & Biases)
* Model registry stages
* Hyperparameter tuning
* Reproducibility checklist (commit, data digest, env lock, artifact link)
* Adaptation decision framework: prompting vs RAG vs PEFT (LoRA) vs full fine-tuning
#### Live Demo: Build a RAG pipeline that answers from company docs with citations
<img width="508" height="566" alt="image" src="https://github.com/user-attachments/assets/96e8ae93-ad33-4e76-8f45-0b1a789fcea7" />

### Serving, Inference, and Optimization
#### Serving Architecture
* Dedicated inference services
* Serving frameworks (Triton, TorchServe, Ray Serve, vLLM, SGLang)
* Prefill vs decode; memory-bound vs compute-bound inference
* Topologies like single model, router/gateway, cascades with fallbacks
* Self-host vs managed APIs (rate limits, data residency, and unit economics)
#### Optimization Levers
* Profiling and baselines like p50/p95 latency
* Semantic caching, dynamic vs continuous batching, autoscaling signals
* Quantization (post-training, quantization-aware, INT8/INT4)
* Mixed precision (FP16/BF16)
* Pruning (structured, unstructured); distillation
* Runtimes and kernels (ONNX Runtime, TensorRT, OpenVINO)
* KV Cache, FlashAttention and speculative decoding
* Packaging (checkpoint/ONNX → service image, gRPC contracts, etc.)
#### Live Demo: Make the assistant faster and cheaper, proven with benchmarks
* Serving, Inference, and Optimization diagram
<img width="518" height="658" alt="image" src="https://github.com/user-attachments/assets/d6806f01-105a-4076-a8c5-0ee24cd0e2a7" />

### Evaluation and Monitoring
#### Evaluating Production AI
* Offline vs online evaluation
* Classification metrics (precision, recall, F1, ROC-AUC vs PR-AUC)
* Threshold selection with cost-based sweeps
* Calibration (reliability diagrams, Brier score, Platt scaling, isotonic regression)
* Eval harnesses (RAGAS, DeepEval), LLM-as-judge calibration
* Multi-turn continuity checks
* Red-team prompts in regression packs
#### Observability and Drift
* Tracing and metrics (OpenTelemetry, Prometheus, Grafana)
* Drift classes: data, concept, embedding, prompt
* Outcome-based alerting
* Continuous training vs CI/CD
* Progressive delivery: shadow → canary → full and rollback gates
* Prompt registries
#### Live Demo: Build eval dashboards and a gate that blocks bad releases
* Evaluation and Monitoring diagram
<img width="502" height="550" alt="image" src="https://github.com/user-attachments/assets/c6af2976-be0e-4791-b93f-ea748b782531" />

### Security and Governance
#### Threats, Authentication, and Privacy
* Threat landscape
* OWASP LLM Top 10
* API authentication (tokens, JWT)
* PII detection and redaction (prompts, logs, responses)
* Encryption in transit and at rest
* Vector store protection
#### Guardrails and Prompt Registry
* Layered defense: input/output filters, structured-output contracts, audit logging
* Groundedness checks
* Model cards, datasheets, fairness audits
* Explainability and citations as RAG explanations
* Regulatory context (EU AI Act, NIST AI RMF)
* Prompt/policy registry: versioning, review, offline checks, release, rollback
#### Live Demo: Lock down the assistant with auth, role-based retrieval, and audited policy releases
* Security and Governance diagram
<img width="505" height="484" alt="image" src="https://github.com/user-attachments/assets/780d58b0-34ec-4d18-a2d5-c53b45fb0149" />

### Scaling
#### Platform Architecture and Release Engineering
* Kubernetes (EKS, GKE, AKS) and ECS; managed ML platforms (Vertex AI, SageMaker, Azure ML)
* Workflow orchestration (Airflow, Prefect, Dagster, Kubeflow, Ray)
* GPU scheduling, node pools, autoscaling, capacity planning
* Canary and blue-green rollouts; auto-rollback gates
* SLOs (latency, error rate, groundedness)
* Incident response and on-call basics
#### Cost Tradeoffs and Case-Study
* Planner/worker patterns
* HITL approval gates; agent tracing; when not to use agents
* Case study: recommendation and search
* Case study: fintech risk scoring
* Case study: consumer assistants
#### Live Demo: Deploy the full platform with a canary release, a cost report, and a human-supervised agent
* Scaling diagram
<img width="502" height="590" alt="image" src="https://github.com/user-attachments/assets/8c2f6f6a-2e74-4c85-a731-5c2d77fa8ce1" />
