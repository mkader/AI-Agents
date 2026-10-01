## What is an AI Agent?
* Generate a report on the latest trends in AI research. If you use a standard LLM, you might:
  1. Ask for a summary of recent AI research papers.
  2. Review the response and realize you need sources.
  3. Obtain a list of papers along with citations.
  4. Find that some sources are outdated, so you refine your query.
  5. Finally, after multiple iterations, you get a useful output.
  <img width="540" height="288" alt="image" src="https://github.com/user-attachments/assets/698d2477-9836-48f6-9761-be917726616f" />

* This iterative process takes time and effort, requiring you to act as the decision-maker at every step.

* How AI agents handle this differently:
  * A Research Agent autonomously searches and retrieves relevant AI research papers from arXiv, Semantic Scholar, or Google Scholar.
    <img width="510" height="203" alt="image" src="https://github.com/user-attachments/assets/6a4003ae-78e3-476e-a867-e5c6e264a932" />
      * A Filtering Agent scans the retrieved papers, identifying the most relevant ones based on citation count, publication date, and keywords.
          <img width="551" height="150" alt="image" src="https://github.com/user-attachments/assets/dbde5a0c-a6d4-4dda-8277-57ee7eacb257" />
      * A Summarization Agent extracts key insights and condenses them into an easy-to-read report.
          <img width="552" height="132" alt="image" src="https://github.com/user-attachments/assets/c7e6da38-945b-447b-94a5-95b6a19c377a" />
      * A Formatting Agent structures the final report, ensuring it follows a clear, professional layout.
          <img width="577" height="132" alt="image" src="https://github.com/user-attachments/assets/e48180ee-e0f1-4e52-aa71-a7965e2dedaa" />

* AI agents not only execute the research process end-to-end but also self-refine their outputs, ensuring the final report is comprehensive, up-to-date, and well-structured - all without requiring human intervention at every step.
  <img width="472" height="412" alt="image" src="https://github.com/user-attachments/assets/e069d99e-e4c1-4a90-9f05-f35d3b9784ff" />

* AI Agents are autonomous systems that can reason, think, plan, figure out the relevant sources and extract information from them when needed, take actions, and even correct themselves if something goes wrong.

## Agent vs LLM vs RAG
* LLM is the brain.
* RAG is feeding that brain with fresh information.
* An agent is the decision-maker that plans and acts using the brain and the tools.

### LLM (Large Language Model)
* An LLM like GPT-4 is trained on massive text data.
* It can reason, generate, summarize but only using what it already knows (i.e., its training data).
* It’s smart, but static. It can’t access the web, call APIs, or fetch new facts on its own.

### RAG (Retrieval-Augmented Generation)
* RAG enhances an LLM by retrieving external documents (from a vector DB, search engine, etc.) and feeding them into the LLM as context before generating a response.
* RAG makes the LLM aware of updated, relevant info without retraining.

### Agent
* An Agent adds autonomy to the mix.
* It doesn’t just answer a question—it decides what steps to take: Should it call a tool? Search the web? Summarize? Store info?
* An Agent uses an LLM, calls tools, makes decisions, and orchestrates workflows just like a real assistant.

## Building blocks of AI Agents
* 6 essential building blocks that make AI agents more reliable, intelligent, and useful in real-world applications:
  1. Role-playing
  2. Focus
  3. Tools
  4. Cooperation
  5. Guardrails
  6. Memory

### 1) Role-playing
* Boost an agent’s performance is by giving it a clear, specific role.
* A generic AI assistant may give vague answers. But define it as a “Senior contract lawyer,” and it responds with legal precision and context.
* Why? Because role assignment shapes the agent’s reasoning and retrieval process. The more specific the role, the sharper and more relevant the output.
  <img width="522" height="173" alt="image" src="https://github.com/user-attachments/assets/33c26f7b-24c5-4612-be25-885f3407526c" />

### 2) Focus/Tasks
* Focus is key to reducing hallucinations and improving accuracy.
* Giving an agent too many tasks or too much data doesn’t help - it hurts.
* Overloading leads to confusion, inconsistency, and poor results.
* For example, a marketing agent should stick to messaging, tone, and audience not pricing or market analysis.
* Instead of trying to make one agent do everything, a better approach is to use multiple agents, each with a specific and narrow focus.
* Specialized agents perform better - every time.
  <img width="492" height="268" alt="image" src="https://github.com/user-attachments/assets/f0bef388-fc5f-4709-9e2f-57ee647bf16b" />

### 3) Tools
* Agents get smarter when they can use the right tools.
* But more tools ≠ better results.
* For example, an AI research agent could benefit from:
  * A web search tool for retrieving recent publications.
  * A summarization model for condensing long research papers.
  * A citation manager to properly format references.
  <img width="532" height="368" alt="image" src="https://github.com/user-attachments/assets/f4bedd3f-9a5a-4d37-b2e7-c326f54ce8c0" />

* But if you add unnecessary tools—like a speech-to-text module or a code execution environment—it could confuse the agent and reduce efficiency.
  
#### 3.1) Custom tools
* While LLM-powered agents are great at reasoning and generating responses, they lack direct access to real-time information, external systems, and specialized computations.
* Tools allow the Agent to:
  * Search the web for real-time data.
  * Retrieve structured information from APIs and databases.
  * Execute code to perform calculations or data transformations.
  * Analyze images, PDFs, and documents beyond just text inputs.
* CrewAI supports several tools that you can integrate with Agents, as depicted below:
  <img width="511" height="515" alt="image" src="https://github.com/user-attachments/assets/e14fe7a7-7fc1-488f-a661-923a3d1e421e" />

* In this example, we're building a real-time currency conversion tool inside CrewAI.
  * Instead of making an LLM guess exchange rates, we integrate a custom tool that fetches live exchange rates from an external API and provides some insights.
  * how you can build one for your custom needs in the CrewAI framework.
    1. installed the tools package ``` pip install crewai-tools
    2. get api key - https://www.exchangerate-api.com/
    3. standard import statements:

Next, we define the input fields the tool expects using Pydantic.

14

DailyDoseofDS.com

Now, we define the CurrencyConverterTool by inheriting from BaseTool:

Every tool class should have the _run method which we will execute whenever
the Agents wants to make use of it.
For our use case, we implement it as follows:

In the above code, we fetch live exchange rates using an API request. We also
handle errors if the request fails or the currency code is invalid.
Now, we define an agent that uses the tool for real-time currency analysis and
attach our CurrencyConverterTool, allowing the agent to call it directly if needed:

15

DailyDoseofDS.com

We assign a task to the currency_analyst agent.

Finally, we create a Crew, assign the agent to the task, and execute it.

Printing the response, we get the following output:

Works as expected!

16

DailyDoseofDS.com

#3.2) Custom tools via MCP
Now, let’s take it a step further.
Instead of embedding the tool directly in every Crew, we’ll expose it as a reusable
MCP tool—making it accessible across multiple agents and flows via a simple
server.
First, install the required packages:

We’ll continue using ExchangeRate-API in our .env file:

We’ll now write a lightweight server.py script that exposes the currency converter
tool. We start with the standard imports:

Now, we load environment variables and initialize the server:

17

DailyDoseofDS.com

Next, we define the tool logic with @mcp.tool():

This function takes three inputs—amount, source currency, and target
currency—and returns the converted result using the real-time exchange rate
API.
To make the tool accessible, we need to run the MCP server. Add this at the end
of your script:

18

DailyDoseofDS.com

This starts the server and exposes your convert_currency tool at:
http://localhost:8081/sse.
Now any CrewAI agent can connect to it using MCPServerAdapter. Let’s now
consume this tool from within a CrewAI agent.
First, we import the required CrewAI classes. We’ll use Agent, Task, and Crew
from CrewAI, and MCPServerAdapter to connect to our tool server.

Next, we connect to the MCP tool server. Define the server parameters to
connect to your running tool (from server.py).

Now, we use the discovered MCP tool in an agent:
This agent is assigned the convert_currency tool from the remote server. It can
now call the tool just like a locally defined one.

19

DailyDoseofDS.com

We give the agent a task description:

Finally, we create the Crew, pass in the inputs and run it:

Printing the result, we get the following output:

20

DailyDoseofDS.com

4) Cooperation
Multi-agent systems work best when agents collaborate and exchange feedback.
Instead of one agent doing everything, a team of specialized agents can split tasks
and improve each other’s outputs.

Consider an AI-powered financial analysis system:
● One agent gathers data
● another assesses risk,
● a third builds strategy,
● and a fourth writes the report
Collaboration leads to smarter, more accurate results.
The best practice is to enable agent collaboration by designing workflows where
agents can exchange insights and refine their responses together.

21

DailyDoseofDS.com

5) Guardrails
Agents are powerful but without constraints, they can go off track. They might
hallucinate, loop endlessly, or make bad calls.
Guardrails ensure that agents stay on track and maintain quality standards.

Examples of useful guardrails include:
● Limiting tool usage: Prevent an agent from overusing APIs or generating
irrelevant queries.
● Setting validation checkpoints: Ensure outputs meet predefined criteria
before moving to the next step.
● Establishing fallback mechanisms: If an agent fails to complete a task,
another agent or human reviewer can intervene.
For example, an AI-powered legal assistant should avoid outdated laws or false
claims - guardrails ensure that.
6) Memory
Finally, we have memory, which is one of the most critical components of AI
agents.

22

DailyDoseofDS.com
Without memory, an agent would start fresh every time, losing all context from
previous interactions. With memory, agents can improve over time, remember
past actions, and create more cohesive responses.

Different types of memory in AI agents include:
● Short-term memory – Exists only during execution (e.g., recalling recent
conversation history).
● Long-term memory – Persists after execution (e.g., remembering user
preferences over multiple interactions).
● Entity memory – Stores information about key subjects discussed (e.g.,
tracking customer details in a CRM agent).
For example, in an AI-powered tutoring system, memory allows the agent to
recall past lessons, tailor feedback, and avoid repetition.
