Picking the model is the easiest part of building an AI agent.

The 9 layers around it decide whether it works in production:

1. Harness
The loop that lets a model act. Context in, tool call out, result back, repeat.

2. Memory & state
What it remembers between runs, and where it is in the task right now.

3. RAG (retrieval-augmented generation)
Pulls the right docs into the prompt so answers come with sources.

4. MCP (Model Context Protocol)
One standard plug for tools and data. Makes it easier for data to be processed. 

5. Skills
Reusable know-how in a SKILL.md file. Only the name and description load until a task needs it.

6. Guardrails
Permissions, sandboxes and a human sign-off before anything risky.

7. Evals
Scores outputs against what you expected, before your users do it for you and get disappointed.

8. A2A (Agent2Agent)
How agents from different vendors find each other and hand off work.

9. Multi-agent
An orchestrator splits the job. Specialist agents run in parallel.
MCP, A2A and Agent Skills are all open standards now, so what you build on them isn't stuck with one vendor.

<img width="800" height="999" alt="9 ai agent terms" src="https://github.com/mkader/AI-Agents/blob/main/notes/9%20ai%20agent%20terms.jpg" />
