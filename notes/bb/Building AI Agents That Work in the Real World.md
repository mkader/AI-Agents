# Building AI Agents That Work in the Real World
<img width="334" height="374" alt="image" src="https://github.com/user-attachments/assets/4e34c353-a572-409e-81bd-d3bf3ddd2d76" />
<img width="616" height="729" alt="image" src="https://github.com/user-attachments/assets/ddc7477c-5350-4f13-9c40-74855d70cc05" />

## Part 1 - From LLM to Agent
* What an LLM can and can't do on its own
* Tool calling (tool definitions, tool calls, tool results)
* The agent loop (ReAct, stopping when there are no tool calls)
* Retrieval for answers grounded in your docs
* Agent = model + harness
<img width="712" height="332" alt="image" src="https://github.com/user-attachments/assets/755c1cd4-905e-40a0-adfb-36cea55c7893" />

## Part 2 - Agents in Production
* What changes at scale (more tools, longer runs, bigger knowledge bases)
* Why a demo harness fails in production
* Design of a production harness (loop, tools, context, memory, retrieval)
<img width="782" height="302" alt="image" src="https://github.com/user-attachments/assets/433cac82-9a98-4d49-bef0-47dd1f63d6b6" />

## Part 3 - The Agent Loop
* Prompt caching and exact prefix matching
* Stable content before volatile content
* Append-only updates
<img width="421" height="389" alt="image" src="https://github.com/user-attachments/assets/d239e85f-c079-4e22-8371-4306f248b693" />

## Part 4 - Tools at Scale
* Tool overload and ambiguity
* Curated tool registries (merge, drop, rewrite descriptions)
* Dynamic tool loading
<img width="786" height="352" alt="image" src="https://github.com/user-attachments/assets/21df0270-6745-4145-bcd5-2f8820f40aa6" />

## Part 5 - Context Management
* Bounding tool output (size caps, head-and-tail truncation)
* Clearing stale tool results
* Compaction (summarization vs. sliding window)
<img width="562" height="394" alt="image" src="https://github.com/user-attachments/assets/4e0afaa0-bf9c-4a93-8d48-485d07fd8f72" />

## Part 6 - Agent Memory
* Single-file memory and fact extraction
* Dreaming: offline consolidation (merge, replace, expire)
* From notes to a linked wiki
* A small index loaded every session, with notes read on demand
<img width="672" height="384" alt="image" src="https://github.com/user-attachments/assets/cdd2235e-88b9-473e-abf2-3c8a4f072812" />

## Part 7 - Retrieval
* One-shot RAG and where it fails
* Agentic search inside the loop (just-in-time retrieval)
<img width="494" height="402" alt="image" src="https://github.com/user-attachments/assets/1af8eaf2-2445-498d-b3f9-9616d0635674" />

## Part 8 - The Production Harness, End to End
* The whole harness, in one picture
* A tip for each part
<img width="784" height="286" alt="image" src="https://github.com/user-attachments/assets/5b2c73c3-cbcc-4eda-b345-a470bee1ad91" />

* https://live.bytebytego.com/masterclasses/build-ai-agents

