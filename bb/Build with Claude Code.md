# Build with Claude Code
<img width="390" height="322" alt="image" src="https://github.com/user-attachments/assets/f0b880bf-7081-4c9f-a406-bb6fc6a02343" />

## What you'll learn

### Understand how Claude Code works
* Master the agentic loop: gather, act, verify
* Navigate with keyboard shortcuts and slash commands
* Control permissions from safe mode to full autonomy

### Master context engineering
* Write a CLAUDE.md under 200 lines that transforms Claude's output
* Watch AI fail, diagnose why, and fix it with rules
* Build a Second Brain that persists knowledge across sessions

### Automate workflows with Skills
* Turn any repeatable workflow into a one-line command
* Compose Skills that call other skills
* Build a suite of 3-4 reusable skills for your team

### Connect Claude to external tools
* Install and scope MCP servers for real data sources
* Automate the browser with /chrome screenshots and console reads
* Build self-correcting chains that catch and fix their own bugs

### Run parallel AI agents
* Manage 2-3 Claude instances at once with git worktrees
* Choose between subagents, agent teams, and separate instances
* Build a full-stack app in parallel in a single assignment

### Get out of the loop
* Audit your codebase for AI-readiness with the Five Pillars
* Wire hooks so agents validate work unattended
* loops, goals, and graph engineering
* Ship a real feature where agents execute and you review the PR

## Claude Code Fundamentals
### Claude Code Basics
* The agentic loop (gather → act → verify)
* Installation and your first interactive session
* Slash commands and keyboard shortcuts
* The permission system (Default, acceptEdits, plan, dontAsk, bypassPermissions)
### Context Engineering
* The context window as a finite working memory
* The "fresh and condensed" principle
* Context management commands
* Layers of memory: CLAUDE.md, Second Brain, Auto Memory
* Lazy loading
### Automation with Skills
* Skills as reusable workflows and the anatomy of a SKILL.md
* Skills creation vs. Built-in skills
* Composability (skills that call other skills)
* Claude Code Fundamentals diagram
<img width="1086" height="537" alt="image" src="https://github.com/user-attachments/assets/47bcdfd6-3c6e-4e1b-9a51-e977aeb1fc02" />
<img width="912" height="509" alt="image" src="https://github.com/user-attachments/assets/1a9ba160-10a9-468c-9030-d7941932c71a" />
<img width="1109" height="480" alt="image" src="https://github.com/user-attachments/assets/d5991e73-4538-4988-8547-cbcf09085f59" />

## Practice the Foundations
### Onboard to a New Codebase
* Point Claude Code at an unfamiliar repo
* Use plan mode to explore before acting
* Produce a useful artifact: roast, architecture doc, onboarding guide, or test gap analysis
* Verification
### CLAUDE.md (the mesh walkthrough)
* Initialize CLAUDE.md
* Write canonical patterns and migrate the legacy code
* Build a search feature from a senior-engineer spec on clean conventions
* Translate a designer email (violet/lime accents) into a durable rule
### Skill from a real workflow (log triage)
* Take stock of a UI-only log archive and identify the automation gap
* Add an API endpoint to expose the tool to Claude
* Author a markdown report template using documented placeholders
* Run the pipeline manually: curl → parse → filter → categorize → aggregate → render
* Wrap the entire flow as a new skill: /triage-logs <log-id>
* Run the skill on a fresh log with no re-prompting
* Practice the Foundations diagram
<img width="1037" height="482" alt="image" src="https://github.com/user-attachments/assets/7369f15b-f01b-4ea2-b687-47bef1aa0e48" />
<img width="1080" height="578" alt="image" src="https://github.com/user-attachments/assets/f99d52a1-f9d9-4a3d-b257-c51b39be9afc" />
<img width="1006" height="400" alt="image" src="https://github.com/user-attachments/assets/de3fdb51-9844-4714-9b7b-e60efcb973ff" />

## Scaling Up with MCPs, Parallel Agents, and Agentic Engineering
### MCPs, CLIs, and Agentic Tooling
* Model Context Protocol (MCP) and installation
* When NOT to use MCP (token cost, output limits, signal-to-noise)
* Browser automation with /chrome (navigate, screenshot, read console, record GIFs)
* The self-correcting chain with no human in the middle (build → screenshot → detect → fix → verify)
* Agentic engineering
* Hook lifecycle
### Parallel Development and Agentic Validation
* Git worktrees (claude --worktree) for safe parallel isolation
* The multi-agent workflow (managing a team of Claudes)
* Subagents (centralized, lower-cost) vs Agent Teams (peer-to-peer, higher-cost)
* Notification hooks (Notification, Stop, SubagentStop, StopFailure)
* /chrome in parallel workflows for automated visual regression
### The Five Pillars of Agentic Engineering
* Common challenges at scale (context bloat, skill explosion, review bottleneck, friction points, evaluation gaps, you-as-the-bottleneck)
* Techniques to build at scale
* Compound engineering
* Auditing your codebase for AI-readiness
* Designing harnesses to eliminate friction
* Scaling Up with MCPs, Parallel Agents, and Agentic Engineering diagram
<img width="1088" height="442" alt="image" src="https://github.com/user-attachments/assets/5d1df70b-e8f1-4d32-93e6-b33fb1f26589" />
<img width="1033" height="582" alt="image" src="https://github.com/user-attachments/assets/21e78535-cd0f-40cd-9b31-f3b469ba8978" />
<img width="1066" height="514" alt="image" src="https://github.com/user-attachments/assets/29741a4d-1e6d-4b69-81a3-d867d10b7447" />

## Build at Scale
### Connect Claude to the Real World with MCPs
* Install MCP servers (Figma, Blender, Slack, or Notion)
* Build a tool that reads from and writes to an external system via MCP
* Add validation hooks that check the tool's output automatically
* Use /chrome to visually verify the built interface or output
### Build a Full-Stack App in Parallel
* Set up a shared CLAUDE.md with stack, ports, and API contract
* Run Claude instances simultaneously in separate worktrees (backend, frontend, tests)
* Decompose a full-stack app into independent tasks suitable for parallel agents
* Merge worktree branches together, resolving at least one conflict in the process
* Use /chrome to validate the merged frontend renders and functions correctly
### Capstone
* Pick a project you actually want to ship: your own repo, a side project, or one of three starter PRDs (CodeCoach, Korridor, or Wordle iOS). Your stack, your language.
* Write a CLAUDE.md, scope .claude/rules/, and build a Second Brain so Claude never has to ask for context.
* Set up a validation harness with tests, linting, hooks, and /chrome so Claude can self-check and self-correct.
* Wire up agentic tooling, including MCPs, 2–3 reusable skills, and project slash commands that help your agents do more on your behalf.
* Define agent team roles and accelerate your development workflow.
* Build at Scale diagram
<img width="662" height="666" alt="image" src="https://github.com/user-attachments/assets/54979484-7627-486a-9369-67baa3591ae0" />

* https://live.bytebytego.com/courses/claude-code?promotionCode=INSIDER20
