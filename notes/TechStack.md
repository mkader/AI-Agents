[CrewAI](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#crewai) | [serper](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#serper) | [firecrawl](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#firecrawl) | [Load Env](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#load-env) | [Pydantic](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#pydantic) | [LightningAI's LitServe](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#lightningais-litserve) | [FastMCP](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#fastmcp-created-by-prefect) | [Ollama](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#ollama) | [CartesiaAI](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#cartesiaai) | [AssemblyAI](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#assemblyai) | [LlamaIndex](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#llamaindex-to-power-rag) | [Livekit](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#livekit-for-orchestration) | [Browserbase](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#browserbase) | [Streamlit](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#streamlit) | [BrightData](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#brightdata) | [Zep](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#zep) | [MS Agent Framework (MAF)](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#ms-agent-framework-maf) | [Motia or III](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#motia-or-iii) | [langchain](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#langchain) | [Pandas](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#Pandas) | [News API](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#News) | [tavily](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#tavily) | [LangGraph](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#LangGraph) | [LangSmith](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#LangSmith) | [Evaluation](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#Evaluation) | [Guardrails](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#Guardrails) | [Agents | Harness](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#Agents--Harness)  | [Harness](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#harness)  | [Runing Agent](https://github.com/mkader/AI-Agents/blob/main/notes/TechStack.md#Runing-Agent)

### Runing Agent
* Agents support both synchronous and asynchronous execution using either .invoke() / await .ainvoke() for full responses, or .stream() / .astream() for incremental streaming output.
	```
	from langchain.agents import create_agent
	agent = create_agent(...)
	response = agent.invoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]}) 
	response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})
	```
* Inputs and outputs
	* Agents use a language model that expects a list of messages as an input. Therefore, agent inputs and outputs are stored as a list of messages under the messages key in the agent state.
	* Input format - Agent input must be a dictionary with a messages key. Supported formats are:
		```
		Format				Example
		String				{"messages": "Hello"} — Interpreted as a HumanMessage
		Message dictionary	{"messages": {"role": "user", "content": "Hello"}}
		List of messages	{"messages": [{"role": "user", "content": "Hello"}]}
		With custom state	{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"} — If using a custom state_schema
		```
	* Output format - Agent output is a dictionary containing:
		* messages: A list of all messages exchanged during execution (user input, assistant replies, tool invocations).
		* Optionally, structured_response if structured output is configured.
		* If using a custom state_schema, additional keys corresponding to your defined fields may also be present in the output. These can hold updated state values from tool execution or prompt logic.
* Streaming output - Agents support streaming responses for more responsive applications. This includes:
	* Progress updates after each step
	* LLM tokens as they're generated
	* Custom tool messages during execution
	```
	async for chunk in agent.astream(
	for chunk in agent.stream(
	    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
	    stream_mode="updates"
	):
	    print(chunk)
	```
* Max iterations
	* To control agent execution and avoid infinite loops, set a recursion limit.
	* This defines the maximum number of steps the agent can take before raising a GraphRecursionError.
 	* You can configure recursion_limit at runtime or when defining agent via .with_config(): :::
	* "Runtime"
		```
		from langgraph.errors import GraphRecursionError
		
		max_iterations = 3
		recursion_limit = 2 * max_iterations + 1
		
		try:
		    response = agent.invoke(
		        {"messages": [{"role": "user", "content": "what's the weather in sf"}]},
		        {"recursion_limit": recursion_limit},
		    )
		except GraphRecursionError:
		    print("Agent stopped due to max iterations.")
		```
	* ".with_config()"
		```
		agent_with_recursion_limit = agent.with_config(recursion_limit=recursion_limit)

  		response = agent_with_recursion_limit.invoke(
			{"messages": [{"role": "user", "content": "what's the weather in sf"}]},
		)
		```

### [Agents](https://docs.langchain.com/oss/python/langchain/agents) | Harness
* An agent is a model calling tools in a loop until a given task is complete.
	<img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/92918a8c-be9d-4cce-9e7a-2984ee3c3293" />
* A harness is everything around that loop: the prompt, the tools, and any middleware that shapes the model’s behavior. ``` Agent = Model + Harness ```
* The job of a harness: get the model the right context at the right time for the given task.
	```
	from langchain.agents import create_agent

	def check_weather(location: str) -> str:
	    '''Return the weather forecast for the specified location.'''
	    return f"It's always sunny in {location}"

 	class Answer(BaseModel):
    	summary: str
    	confidence: float

 	class MyState(AgentState):
    	user_id: str
    	call_count: int
 
	agent = create_agent(
	    model="anthropic:claude-sonnet-4-5-20250929",
	    tools=[check_weather],
	    system_prompt="You are a helpful assistant",
 		response_format=Answer,
 		state_schema=MyState,
	)
	inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}

 	result = agent.invoke(
	    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
	    config=config,
	)

 	for chunk in agent.stream(inputs, stream_mode="updates"):
	    print(chunk)
 
 	stream = agent.stream_events(
	    {"messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]},
	    version="v3",
	)
	for snapshot in stream.values:
	    # Each snapshot contains the full state at that point
	    latest_message = snapshot["messages"][-1]
	```
 <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/f1982ef2-4eb3-4e77-9eba-5951b373a3d8" />
 
* Core Components
	* Model, tools, system_prompt
 	* Structured output - Return a validated schema from the agent using response_format=
 	* Agent state - Every agent manages its execution context through AgentState, a typed dictionary that holds the current conversation history and any custom fields your tools and middleware need.
  		* For a weather agent, the state might look like this: ``` state = {"city": "Seattle", "units": "fahrenheit", "weather_result": None, "step": "need_weather" } ```
		* The agent can update it as it works: 1) It reads the request: “What’s the weather in Seattle?” 2)It stores city = "Seattle" and units = "fahrenheit". 3) It calls the weather tool and saves the result in weather_result. 4) It uses that result to answer the user.
		* State can include conversation history, task progress, and tool results.
  		* Some state exists only during one request; other state may be saved between conversations.
    	* The harness manages where state is stored and what is passed to the model.
* Invocation - invoke an agent with a message. Behind the scenes that passes an update to the agent’s State.
	* All agents include a sequence of messages in their state; to invoke the agent, pass a new message along with a thread_id so the agent can persist and resume conversation history
* Streaming - invoke returns the final response at the end of a run.
	* If an agent executes multiple tool calls, users often need progress updates before completion.
	* Use streaming to surface intermediate messages and tool activity as they happen.
* Configure the harness
	* LangChain [create_agent](https://reference.langchain.com/python/langchain/agents/factory/create_agent) is a highly configurable harness or extensible.
 
 	<img width="750" height="200" alt="image" src="https://github.com/user-attachments/assets/2d467edc-7955-47b0-9d7d-9d48b96eab8f" />
	
	* Execution environment
 		* Agents are especially useful when they can take action rather than just generate text.
   		* The execution environment gives the agent a workspace: tools it can call, a filesystem for reading and writing files across turns, and code execution for running scripts or shell commands.
     	```
		from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware, SkillsMiddleware, SummarizationMiddleware
		# Execution environment
		agent = create_agent(
		    model="openai:gpt-5.5",
		    tools=[search],
		    middleware=[FilesystemMiddleware(backend=StateBackend())],
		)
  		```
     * Context management
		* Every model call has a fixed context window.
     	*  As an agent runs, that window fills with accumulating history, tool results, and intermediate steps.
      	*  Summarization compresses history before overflow hits; memory loads persistent instructions at startup so knowledge carries across sessions; skills surface domain knowledge on demand rather than loading everything upfront.
		```
	     backend = StateBackend()
		    middleware=[
		        FilesystemMiddleware(backend=backend),
		        SummarizationMiddleware(model=model, backend=backend),
		        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
		        SkillsMiddleware(backend=backend, sources=["./skills/"]),
		    ],
     	```
  	* Planning and delegation
		* Complex tasks often exceed what one context window can handle.
  		* Delegation lets the main agent break work into pieces, hand them to subagents that each run in their own isolated context, and stay focused on coordination rather than execution.
    	* Work can run in parallel; the main agent’s context stays clean.
		```	
		@tool
		def search(query: str) -> str:
		    """Search for a query and return a short summary."""
		    return f"Search results for: {query}"

	    middleware=[
	        FilesystemMiddleware(backend=backend),
	        TodoListMiddleware(),
	        SubAgentMiddleware(
	            backend=backend,
	            subagents=[
	                {
	                    "name": "researcher",
	                    "description": "Searches and returns a structured summary.",
	                    "system_prompt": "Use the search tool to research the question and summarize key points.",
	                    "tools": [search],
	                    "model": "anthropic:claude-sonnet-4-6",
	                    "middleware": [],
	                }
	            ],
	        ),
	    ],
		```
  	* Fault tolerance
		* Agents in production encounter failures that rarely appear in development: rate limits, model timeouts, transient API errors.
  		* Fault tolerance middleware handles these at the infrastructure level so your tools and business logic don’t need try/catch around every call.
	    ```
	    middleware=[
	        ModelRetryMiddleware(max_retries=3),
	        ToolRetryMiddleware(max_retries=2),
	    ],
	    ```
	* Guardrails
 	* Steering
		* Full autonomy isn’t always appropriate.
  		* Steering lets you place humans at specific decision points—before destructive writes, expensive API calls, or anything requiring judgment—without restructuring your agent.
    	* The agent pauses and waits; a human approves, edits, or rejects; execution continues.
    	``` middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_file": True})], ```

### Harness
* An AI agent harness is the software around an AI model that lets it do more than just generate text.
* The harness receives a request, gives the model instructions and available tools, runs approved tool calls, sends their results back to the model, and returns the final answer.
* Think of the model as the decision-maker and the harness as the operator that connects it to tools and enforces the rules.
* Example: a weather assistant
	* A user asks: “What’s the weather in Seattle?”
	1. The harness sends the request and instructions to the model, along with a description of a weather lookup tool.
	2. The model decides it needs current weather and requests that tool.
	3. The harness checks the request, calls the weather service, and gets the result.
	4. The harness gives that result to the model.
	5. The model responds: “It’s 54°F and raining in Seattle.”
* Simplified pseudocode:
	```
	def handle_request(user_message):
	    messages = [
	        {"role": "system", "content": "Answer helpfully. Use the weather tool for current conditions."},
	        {"role": "user", "content": user_message},
	    ]
	
	    response = model.respond(messages, tools=[weather_tool])
	
	    if response.tool_call:
	        # The harness runs the requested tool and returns its result to the model.
	        result = weather_tool(response.tool_call.arguments)
	        messages.append(response.tool_call)
	        messages.append({"role": "tool", "content": result})
	        response = model.respond(messages, tools=[weather_tool])
	
	    return response.text
	```  
* The harness is responsible for things like tool execution, permissions, conversation state, error handling, and limits. The model chooses what it thinks should happen; the harness determines what actually happens.

### Guardrails
* Guardrails are checks and rules around an AI system that help keep its inputs, behavior, and outputs within safe and useful boundaries.
* They can protect sensitive information, reduce harmful or off-topic answers, and stop the model from making claims that violate business rules.
* Think of them as several checkpoints:
	* Before the AI responds: Check the request—for example, block requests for someone else’s private account details.
	* While guiding the AI: Give it clear rules and approved information to use.
	* After it responds: Check that the answer is safe and follows policy; block it or provide a safe fallback if it doesn’t.
	* During testing: Run example questions to catch failures before users encounter them.
 * Example: customer-support AI
	* Suppose a support bot should tell customers that returns are allowed within 30 days.
		* A user asks: “Can I return this after 90 days?”
		* The AI might mistakenly answer: “Yes, you have 90 days.”
	* A guardrail checks the answer against the actual return policy. Since it contradicts the policy, the bot shouldn’t send that answer. It can instead say: “The return window is 30 days. For help with an exception, please contact support.”
	* A key point: a prompt that says “always follow the policy” is helpful, but it isn’t a complete guardrail by itself. An output check and a safe fallback provide another layer of protection.
	```
	import os
	from dotenv import load_dotenv
	
	from langchain_core.output_parsers import StrOutputParser
	from langchain_core.prompts import ChatPromptTemplate
	from langchain_openai import ChatOpenAI
	
	load_dotenv()
	
	# Reuse the Azure OpenAI settings already in your .env file.
	llm = ChatOpenAI(
	    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LLM"],
	    base_url=os.environ["AZURE_OPENAI_ENDPOINT_V1"],
	    api_key=os.environ["AZURE_OPENAI_API_KEY"],
	    temperature=0,
	)
	
	prompt = ChatPromptTemplate.from_messages(
	    [
	        (
	            "system",
	            "Answer general customer-support questions clearly. "
	            "Do not invent company policies or account details. "
	            "If you don't know, say so and direct the customer to support.",
	        ),
	        ("human", "{question}"),
	    ]
	)
	
	support_chain = prompt | llm | StrOutputParser()
	
	RETURN_POLICY = "Items can be returned within 30 days of purchase."
	SAFE_FALLBACK = "I couldn't generate an answer. Please contact support."
	
	
	def ask_ai(inputs: dict) -> dict[str, str]:
	    question = inputs["question"]
	    question_lower = question.casefold()
	
	    # For policy questions, return the approved answer rather than generated text.
	    if "return" in question_lower or "refund" in question_lower:
	        return {"answer": RETURN_POLICY}
	
	    answer = support_chain.invoke({"question": question}).strip()
	    return {"answer": answer or SAFE_FALLBACK}
	
	print(ask_ai({"question": "Can I return an item after 90 days?"}))

 	{'answer': 'Items can be returned within 30 days of purchase.'}
 	```
	```
	import os
	
	from dotenv import load_dotenv
	from langchain.agents import create_agent
	from langchain.agents.middleware import PIIMiddleware
	from langchain.tools import tool
	from langchain_openai import ChatOpenAI
	
	load_dotenv()
	
	@tool
	def search(query: str) -> str:
	    """Search the support knowledge base and return a short summary."""
	    return f"Support knowledge base results for: {query}"
	
	
	model = ChatOpenAI(
	    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LLM"],
	    base_url=os.environ["AZURE_OPENAI_ENDPOINT_V1"],
	    api_key=os.environ["AZURE_OPENAI_API_KEY"],
	    temperature=0,
	)
	
	agent = create_agent(
	    model=model,
	    tools=[search],
	    system_prompt=(
	        "You are a customer-support assistant. Use the search tool for support "
	        "questions. Never ask for or reveal sensitive personal information."
	    ),
	    middleware=[
	        PIIMiddleware(
	            "email",
	            strategy="redact",
	            apply_to_input=True,
	            apply_to_output=True,
	        ),
	    ],
	)
	
	
	def ask_agent(question: str) -> str:
	    result = agent.invoke(
	        {"messages": [{"role": "user", "content": question}]}
	    )
	    print(result)
	    return str(result["messages"][-1].content)
	
	
	if __name__ == "__main__":
	    print(ask_agent("Search for the return policy. My email is alice@example.com."))
 	```

	<img width="1027" height="670" alt="image" src="https://github.com/user-attachments/assets/d7a3f149-d8b5-46d6-8394-be27fca011b2" />

### Evaluation
* It's a repeatable way to measure how well a model or agent performs on a set of representative tasks.
* You give it test inputs, assess its outputs against criteria, and track the results—often before and after a prompt, model, or code change.
* Common eval terms:
	* Dataset: The test cases (inputs, and sometimes reference answers).
	* Metric: A measurable score, such as accuracy, pass rate, latency, or cost.
	* Grader: The rule or method that scores an output. It can be deterministic code, a human, or another model (“LLM-as-judge”).
	* Baseline / regression: The existing score and whether a change made performance worse.
	```
	test_cases = [
	    {
	        "question": "When can I return an item?",
	        "answer": "You can return it within 30 days of purchase.",
	        "must_include": ["30 days"],
	    },
	    {
	        "question": "How do I reset my password?",
	        "answer": "Select Forgot password on the sign-in page.",
	        "must_include": ["Forgot password", "sign-in page"],
	    },
	]
	
	def grade(case):
	    answer = case["answer"].lower()
	    missing = [
	        phrase for phrase in case["must_include"]
	        if phrase.lower() not in answer
	    ]
	    return {"passed": not missing, "missing": missing}
	
	results = [grade(case) for case in test_cases]
	pass_rate = sum(r["passed"] for r in results) / len(results)
	
	print(f"Pass rate: {pass_rate:.0%}")
	for case, result in zip(test_cases, results):
	    print(case["question"], result)
	```
	```
 		from openai import OpenAI
		import os
		from dotenv import load_dotenv
		load_dotenv()
		
		client = OpenAI(
		    base_url=os.environ["AZURE_OPENAI_ENDPOINT_V1"],
		    api_key=os.environ["AZURE_OPENAI_API_KEY"]
		)
		test_cases = [
		    {
		        "question": "When can I return an item?",
		        "must_include": ["30 days"],
		    },
		    {
		        "question": "How do I reset my password?",
		        "must_include": ["Forgot password"],
		    },
		    {
		        "question": "When can I return an item?",
		        "must_include": ["90 days"],
		    },
		]
		
		def ask_ai(question: str) -> str:
		    response = client.chat.completions.create(
		        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LLM"],
		        messages=[
		            {
		                "role": "system",
		                "content": (
		                    "Answer customer support questions clearly. "
		                    "Items can be returned within 30 days of purchase."
		                ),
		            },
		            {"role": "user", "content": question},
		        ],
		    )
		    return response.choices[0].message.content or ""
		
		def grade(answer: str, required_phrases: list[str]) -> bool:
		    return all(phrase.lower() in answer.lower() for phrase in required_phrases)
		
		results = []
		for case in test_cases:
		    answer = ask_ai(case["question"])
		    passed = grade(answer, case["must_include"])
		    results.append(passed)
		    print(f"Q: {case['question']}\nA: {answer}\nPassed: {passed}\n")
		
		print(f"Pass rate: {sum(results) / len(results):.0%}")
 	```

  	<img width="566" height="530" alt="image" src="https://github.com/user-attachments/assets/e76703ae-8987-43e8-bf6a-173da8a6aae8" />

#### [LangSmith](https://smith.langchain.com/)
* It's a platform for building, testing, and monitoring AI applications. It works with LangChain and LangGraph, but can also be used with other frameworks.
* Think of it as a flight recorder and test bench for an AI app:
	* Tracing: Records an app’s steps—LLM calls, tool use, inputs, outputs, timing, and errors—so you can see what happened when a response was poor or slow.
	* Evaluation: Lets you test outputs against examples or scoring criteria, and compare changes to prompts or models.
	   	```
	    import os
	
		from dotenv import load_dotenv
		from langsmith import Client, evaluate
		from langsmith.schemas import Example, Run
		from openai import OpenAI
		
		load_dotenv()
		
		openai_client = OpenAI(
		    base_url=os.environ["AZURE_OPENAI_ENDPOINT_V1"],
		    api_key=os.environ["AZURE_OPENAI_API_KEY"],
		)
		
		DATASET_NAME = "support-faq-eval-v1"
		test_cases = [
		    {
		        "question": "When can I return an item?",
		        "must_include": ["30 days"],
		    },
		    {
		        "question": "How do I reset my password?",
		        "must_include": ["Forgot password"],
		    },
		    {
		        "question": "When can I return an item?",
		        "must_include": ["90 days"],
		    },
		]
		
		langsmith_client = Client()
		if not langsmith_client.has_dataset(dataset_name=DATASET_NAME):
		    dataset = langsmith_client.create_dataset(
		        dataset_name=DATASET_NAME,
		        description="Support FAQ evaluation cases, including an intentional failure case.",
		    )
		    langsmith_client.create_examples(
		        dataset_id=dataset.id,
		        examples=[
		            {
		                "inputs": {"question": case["question"]},
		                "outputs": {"must_include": case["must_include"]},
		            }
		            for case in test_cases
		        ],
		    )
		
		
		def ask_ai(inputs: dict) -> dict[str, str]:
		    response = openai_client.chat.completions.create(
		        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LLM"],
		        messages=[
		            {
		                "role": "system",
		                "content": (
		                    "Answer customer support questions clearly. "
		                    "Items can be returned within 30 days of purchase. "
		                    "To reset a password, select Forgot password."
		                ),
		            },
		            {"role": "user", "content": inputs["question"]},
		        ],
		    )
		    return {"answer": response.choices[0].message.content or ""}
		
		
		def grade_required_phrases(run: Run, example: Example) -> dict:
		    if run.outputs is None or example.outputs is None:
		        raise ValueError("The model response and expected phrases are required for grading.")
		
		    answer = run.outputs["answer"].lower()
		    required_phrases = example.outputs["must_include"]
		    missing = [
		        phrase for phrase in required_phrases
		        if phrase.lower() not in answer
		    ]
		    return {
		        "key": "required_information",
		        "score": not missing,
		        "comment": (
		            f"Missing required phrase(s): {', '.join(missing)}"
		            if missing
		            else "All required phrases were present."
		        ),
		    }
		
		
		results = evaluate(
		    ask_ai,
		    data=DATASET_NAME,
		    evaluators=[grade_required_phrases],
		    client=langsmith_client,
		    experiment_prefix="support-faq-eval",
		    description="Check whether support answers include required information.",
		)
		> python eval.py
	     View the evaluation results for experiment: 'support-faq-eval-e3eb3c5f' at:
	     https://smith.langchain.com/o/77aba6cc-e7f9-4245-a3b5-36012d9d5606/datasets/37b476bb-daa9-4914-afd2-658fb6da48f2/compare?selectedSessions=f1d3548b-0922-4c10-beaa-2be28a3b1828
	     3it [00:03,  1.31s/it]
	    ```
	    <img width="1672" height="832" alt="image" src="https://github.com/user-attachments/assets/a9dd180a-a1e8-4b54-8b6f-bc41819ffed1" />
	
	  	<img width="1681" height="302" alt="image" src="https://github.com/user-attachments/assets/378119cc-342b-48c4-a783-fe2b9e969bc6" />

	* Prompt management: Helps create, version, and test prompts.
	* Monitoring: Shows how the application behaves in production, including failures and usage patterns.
	* Deployment: Offers tools for deploying certain agent applications and workflows.
* How it relates to LangGraph: LangGraph defines the steps and control flow of your agent; LangSmith helps you inspect, evaluate, and monitor how that agent performs.
* For example, if a support bot gives a bad answer, a LangSmith trace can show whether the issue came from retrieval, a tool call, the prompt, or the model’s response.
* Spot issues and improve the performance of your LangGraph projects. Trace data to debug, test, and monitor your LLM apps built with LangGraph
  
#### [LangGraph](https://www.langchain.com/langgraph)
* It's a framework for building AI applications as graphs—a set of steps connected by paths.
* It’s useful when an AI workflow needs more than a simple prompt-and-response chain, such as branching, looping, remembering state, or pausing for human input.
* A graph has 3 main parts:
	* State: The shared data passed through the workflow, such as the conversation history or tool results.
	* Nodes: Steps that read the state and update it. A node might call an LLM, search documents, or run a tool.
	* Edges: Rules for what runs next. An edge can lead to a fixed next step or choose a route based on the current state.
* For example, a support assistant might:
	* Receive a question.
	* Decide whether it needs a knowledge-base search.
	* Search if needed.
	* Draft an answer.
	* Ask a person to approve it for sensitive cases.
* Unlike a simple linear chain, a graph can loop back—for example, to revise an answer if it doesn’t meet a quality check.
* LangGraph also supports saving workflow state so an application can resume later, which is useful for long-running tasks and conversations.
* In short: LangGraph helps you control the steps and state of an AI workflow, especially when it needs decisions, tools, memory, or human review.

#### [tavily](https://www.tavily.com/)
* Connect your AI agents to the web. One secure API for real-time web access.

#### [News](https://newsapi.org/)
* Search worldwide news with code. Locate articles and breaking news headlines from news sources and blogs across the web with our JSON API

#### [Pandas](https://pandas.pydata.org) 
* It's open-source Python library used for data manipulation, cleaning, and analysis.
* Pandas as like Excel or Google Sheets to work with tables, columns, and rows, built entirely inside Python code. It allows you to load massive datasets, clean messy information, and calculate advanced statistics in just a few lines of code.
* The Two Core Data Structures of organizing data:
	1. Series: A one-dimensional labeled array. Think of it as a single column in a spreadsheet, accompanied by an index (row labels).
	2. DataFrame: A two-dimensional table with rows and columns. This is the most widely used object in Pandas and represents an entire spreadsheet.
* What Can You Do with Pandas? - Load Data (CSV, Exccel, SQL), Clean Data, Filter & Select, Aggregate & Group, Merge & Join
* Code
```
import pandas as pd

df = pd.read_csv("employee_data.csv") # 1. Load a spreadsheet dataset
print(df.head()) # 2. Preview the first 5 rows
df = df.fillna(0) # 3. Clean data by filling empty cells with 0
high_earners = df[df["salary"] > 80000] # 4. Filter for high earners
print(high_earners.describe()) # 5. Get a quick statistical breakdown (mean, min, max, etc.)
```

#### [langchain](https://www.langchain.com/)
* It's an open-source framework that connects LLMs to external data sources, memory, and software tools.
* Instead of writing custom code to plug models into databases or APIs, developers use LangChain as the "plumbing" to build context-aware AI applications.
* What LangChain Is For?, LangChain solves three major limitations of raw LLM APIs:
	* Giving Models Memory: Raw LLM calls forget past messages. LangChain provides built-in memory modules to track and pass conversation history.
	* Connecting to Private Data (RAG): Models only know their training data. LangChain helps pull information from PDFs, SQL databases, or vector stores so the model can answer questions using your specific data.
	* Enabling Tool Use & Agents: Models normally just output text. LangChain lets an AI act as a reasoning engine that chooses and runs external tools—like web searches, calculators, or custom APIs—to complete multi-step tasks.
* Agent = Model + Harness. LangChain provides create_agent: a minimal, highly configurable harness. The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior.
* DeprecationWarning: `langchain-community` is being sunset and is no longer actively maintained 
* Commands
```
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import END, StateGraph
``` 
* Code - create a simple LangChain agent with a custom tool:
```
# 1/8 pip install -qU langchain "langchain[openai]" #2 "langchain[google-genai]"  #3 "langchain[anthropic]" #4 langchain-openrouter #5 langchain-fireworks #6 langchain-baseten
#7 langchain-ollama #9 langchain-aws #10  "langchain[huggingface]"

import os  
from dotenv import load_dotenv
from langchain.agents import create_agent
#8 azure_openai:gpt-5.5
from langchain.chat_models import init_chat_model

load_dotenv()

def get_weather(city: str) -> str:
	"""Get weather for a given city."""
	return f"It's always sunny in {city}!"

#8 azure_openai:gpt-5.5
model = init_chat_model(
	"azure_openai:gpt-4.1",
	azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME_LLM"],
	api_key=os.environ["AZURE_OPENAI_API_KEY"],
	api_version=os.environ["AZURE_OPENAI_API_VERSION"],
	azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_LLM"],
)

agent = create_agent(
	#model="openai:gpt-5.5" | #1 "google_genai:gemini-3.5-flash-lite", #2 working, got key | "claude-sonnet-4-6", #3 | "openrouter:anthropic/claude-sonnet-4-6", #4
	"fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",#5 | "baseten:zai-org/GLM-5.2", #6 working, got key | "ollama:llama3.2:latest", #7 working, local
	model, #8 working, got key  | "bedrock_converse:us.anthropic.claude-sonnet-4-6", #9 | "huggingface:microsoft/Phi-3-mini-4k-instruct", #10 working, downloaded, got key
	tools=[get_weather],
	system_prompt="You are a helpful assistant",
)

result = agent.invoke(
	{"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```
 
#### [Motia or III](https://iii.dev/) 
* It'is an open-source, event-driven backend framework designed to replace fragmented backend stacks (like Express, BullMQ, Temporal, node-cron, and LangChain) with a single, cohesive system
  
#### [MS Agent Framework (MAF)](https://github.com/microsoft/agent-framework) 
* is an open, multi-language framework for building production-grade AI agents and multi-agent workflows.
* Microsoft Agent Framework is built for teams taking agents from prototype to production. It provides a consistent foundation for building, orchestrating, and operating agent systems
* [Migrated]MS AutoGen for agent orchestration - AutoGen is a framework for creating multi-agent AI applications that can act autonomously or work alongside humans.
	
#### [Zep](https://www.getzep.com/) 
* is a specialized long-term memory and context layer for AI agents that replaces or enhances basic chat history and static RAG.
* Graphiti: temporally-aware knowledge graph engine that extracts entities, relationships, and facts from conversations, JSON data, and documents.
* Konig: graph database service that powers millions of temporal Context Graphs with provenance tracking and sub-200ms retrieval.

#### [BrightData](https://brightdata.com/)
* Web data infrastructure for humans and AI agents to search, crawl, extract the public web site without getting blocked, rate-limited, or anti-bot systems.

#### [Streamlit](https://streamlit.io/)
* It's an open-source Python framework that lets you turn data scripts into interactive web applications using pure Python code, with no HTML, CSS, or JavaScript required
* What is Streamlit for? - Data Visualization & Dashboards (Display charts, maps), ML Prototyping (demo ML models or AI workflows without building a traditional web stack)
```
pip install streamlit

run 1
streamlit hello

import streamlit as st
import numpy as np

st.title("My Quick Data Dashboard")

# Interactive slider widget
num_points = st.slider("Select number of data points:", 10, 100, 50)

# Generate and display data
data = np.random.randn(num_points)
st.line_chart(data)

run 2
streamlit run app.py
```
 
#### [CrewAI](https://crewai.com/)
* It's an open-source Python framework and enterprise platform that lets you build and coordinate teams of autonomous AI agents
* CrewAI for Agent orchestration
* CrewAI is a Python framework for building applications where AI agents work together on tasks.
* You define agents with roles and tools, give them tasks, then group them into a crew that coordinates the workflow.
* [Tools](https://docs.crewai.com/v1.15.28/en/concepts/tools) Powerful crewAI Tools to build AI Agents
	* FileReadTool - Read and extract data from files ``` from crewai_tools import FileReadTool ``` 
	* CodeInterpreterTool - Execute python code generated by Agent
	* ScrapeWebsiteTool - Extract and read content of website
	* SerperDevTool - Retrieve search results from the internet (Google Search API) , 
	* DirectoryReadTool - List the contents of a directory
	* FirecrawlSearchTool - Search and convert websites to markdown, [firecrawl](http://firecrawl.dev/)
	* BrowserbaseLoadTool - Run, manage and monitor headless browers. [Browserbase](https://www.browserbase.com/)
	* PDFSearchTool, TXTSearchTool, GithubSearchTool - RAG tool for searching in PDFs, text files, Githubs repos

| | SerperDevTool |	FirecrawlSearchTool |
| - | - | - |
| Provider	| Serper, which returns Google search results	| Firecrawl
| Typical result	| Search-result titles, links, and snippets	| Search results and, depending on its implementation, extracted page content
| Useful | when	You want a straightforward Google-style web search	| You want web search as part of a Firecrawl-based extraction or RAG workflow

* code
```
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, FileReadTool 

search_tool = SerperDevTool(n_results=10)

llm_local = LLM(
	model="ollama\command-r",
	temperature=0.7
)
  
llm = LLM(
	model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM"),
	api_key=os.getenv("AZURE_OPENAI_API_KEY"),
	temperature=0.7,
	base_url=os.getenv("AZURE_OPENAI_ENDPOINT_LLM")
)

researcher_agent = Agent(
	role="Researcher",
	goal="Research about the user's query and generate insights",
	backstory="You are a helpful assistant that can answer questions about the document.",
	verbose=True,
	tools=[search_tool], # tools=[mcp_tools["currency_converter"]] call mcp tool
	llm=llm
)

researcher_task = Task(
	description="Research about the user's query and generate insights: {query}",
	expected_output="A concise and informative report about the user's query",
	agent=researcher_agent,
)

# Create Crew
crew = Crew(
	agents=[researcher_agent, content_writer_agent],
	tasks=[research_task, writing_task],
	verbose=True
)

return crew.kickoff(inputs={"topic": topic})
```
	
#### CrewAI & LangChain / LangGraph - Core Differences
* Primary Purpose:
	* LangChain / LangGraph: Built as a low-level, modular orchestration and retrieval toolkit. It acts as a graph-based state machine where you explicitly program nodes, edges, state schemas, and custom loops.
	* CrewAI: Built as a high-level, declarative multi-role agent framework. It treats agents like an organizational team with explicit roles, goals, backstories, and automated task delegation.
* Best Use Cases:
	* LangChain: Best for Retrieval-Augmented Generation (RAG), custom chatbots, document processing, and complex nonlinear deterministic logic (often using its upgrade layer, LangGraph).
	* CrewAI: Best for collaborative, multi-phase content pipelines, research workflows, and business automation where agents hand off work like a human team.
* Ecosystem and Scale:
	* LangChain: Features a massive ecosystem (600+ integrations, LangSmith for monitoring, LangServe for deployment) and a much larger developer community.
	* CrewAI: Features a more focused, fast-growing community with native memory structures and built-in tool interoperability.

#### [firecrawl](http://firecrawl.dev/)
* CrewAI FirecrawlSearchTool - Search and convert websites to markdown, 
* It's an open-source web crawling and scraping engine designed specifically to turn the live web into LLM-ready data.
* Traditional web scrapers return raw HTML full of messy code, navigation bars, cookie banners, and ads.
* Firecrawl bypasses that clutter to deliver highly clean, compressed Markdown or structured JSON that AI models and agents can easily consume. (Token Efficiency)
* Endpoint / Feature	What it Does
	* /scrape	Turns a single URL into clean Markdown, raw HTML, a screenshot, or structures it into JSON.
	* /crawl	Recursively traverses an entire domain or documentation site, automatically following links to scrape every page into a unified dataset.
	* /search	Searches the web like a traditional search engine but returns the full page content of top results instead of just tiny text snippets.
	* /interact	Allows an AI agent to execute live sequential actions on a page (e.g., clicking buttons, scrolling, filling out forms, or logging in) before extraction.
	* /map	Rapidly discovers and enumerates the entire link structure/sitemap of a website.
	* /parse	Converts local static documents—like web-hosted PDFs, DOCX, or Excel sheets—into clean text or LaTeX syntax.
* 💡 Why Developers Use It for AI - Token Efficiency, Handles "The Hard Stuff" Automatically (bypasses rate limit, captchas, bot), Built for Agents (natively supports MCP)
* Code
	```
	from firecrawl import Firecrawl
	app = Firecrawl(api_key="fc-YOUR_API_KEY")
	
	# Scrape a page directly into clean Markdown
	result = app.scrape("https://example.com", formats=["markdown"])
	
	print(result.markdown)
	```
#### [Browserbase](https://www.browserbase.com/)
* CrewAI BrowserbaseLoadTool - Run, manage and monitor headless browers. 
* It's a developer platform that provides cloud-based headless browser infrastructure, built specifically for AI agents, web automations, and scraping at scale
* What It Is For - AI Web Automation (AI agents browse, click, type, and navigate those human-only sites), Bypassing Anti-Bot Systems (CAPTCHA), Scaling Headless Browsers (Instead of Playwright, Puppeteer, or Selenium)

#### [Serper](https://serper.dev/) 
* CrewAI SerperDevTool - Retrieve search results from the internet (Google Search API) , 
* Serper is a fast, low-cost Google Search API (or SERP API) built for developers and applications.
* It lets you send search queries to Google via an API and receive clean, structured JSON data (organic results, knowledge graphs, news, images, maps, and shopping) without scraping Google yourself.
* LangChain Integration
	```
	pip install langchain-community

	import os
	from langchain_community.utilities import GoogleSerperAPIWrapper
	
	# Set your API key
	os.environ["SERPER_API_KEY"] = "your_serper_api_key_here"
	
	# Initialize the search tool
	search = GoogleSerperAPIWrapper()
	
	# Run a query
	results = search.run("What is the latest news about SpaceX Falcon 9?")
	print(results)
	```
* Standard Python (Using requests)
	```
	pip install requests

	import json, os, requests
	from dotenv import load_dotenv
	
	load_dotenv()
	
	url = "https://google.serper.dev/search"
	
	payload = {
		"q": "SpaceX Falcon 9 launch",
		"gl": "us",     # Country code (Geotargeting)
		"hl": "en"      # Interface language
	}
	
	headers = {
		'X-API-KEY': os.environ.get("SERPER_API_KEY"),
		'Content-Type': 'application/json'
	}
	
	response = requests.post(url, headers=headers, json=payload, timeout=30)
	response.raise_for_status()
	
	# Print the complete search response as formatted JSON.
	print(json.dumps(response.json(), indent=2))
	```
  #### Load Env
  ```
  from dotenv import load_dotenv
  load_dotenv()
  import os
  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_LLM")
  ```

#### [Pydantic](https://pydantic.dev/) 
* Data validation and parsing library for Python.
* use Pydantic, define a data structure by creating a class that inherits from BaseModel and declaring fields with standard type hints.
```
from pydantic import BaseModel, EmailStr, Field

# 1. Define the blueprint of your data structure
class UserProfile(BaseModel):
	id: int                              # Required field
	username: str                        # Required field
	email: EmailStr                      # Automatically validates email formatting
	is_active: bool = True               # Optional field with a default value

# 2 define the input fields the tool expects using Pydantic.
class CurrencyConverterInput(BaseModel):
  from_currency: str = Field(..., description="The currency code to convert from (e.g USD)")
  to_currency: str = Field(..., description="The currency code to convert to (e.g EUR)")
  amount: float = Field(..., description="The amount of currency to convert")
```

#### [LightningAI's LitServe](https://lightning.ai/litserve)
* It's an open-source, Pythonic AI model serving engine built on top of FastAPI by the creators of PyTorch Lightning at Lightning AI.
* Core Architecture
	* LitAPI: An abstract base class that lets you write custom inference logic by defining 4 main methods:
		* setup(self, device): Loads models, databases, or resources into memory.
		* decode_request(self, request): Transforms incoming HTTP requests into model inputs.
		* predict(self, x): Executes model inference (supports standard or generator/streaming functions).
		* encode_response(self, output): Converts model outputs back into HTTP responses.
	* LitServer: Manages the underlying production infrastructure, including multi-worker concurrency, request load balancing, and hardware allocation.
* Key Features
	* Built-in Dynamic Batching: Groups incoming requests automatically to maximize GPU throughput and efficiency.
	* Native Streaming: Supports real-time token or frame streaming using Python generator syntax (yield).
	* Multi-Model & Agent Support: Run single models, RAG pipelines, or multi-model chains inside a single server without complex YAML configuration files.
* Code Implmentation - What type of model or pipeline you are trying to deploy (LLM, vision, audio, RAG), Whether you need streaming or dynamic batching
* LLM Code
	```
	# install LitServe via pip: pip install litserve
	import litserve as ls
	from transformers import AutoModelForCausalLM, AutoTokenizer
	import torch
	
	class LLMAPI(ls.LitAPI):
		def setup(self, device):
			# 1. Load the model and tokenizer into memory
			self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
			self.model = AutoModelForCausalLM.from_pretrained(
				self.model_name, 
				dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
			).to(device)
			self.device = device
	
		def decode_request(self, request):
			# 2. Extract the prompt and generation parameters from the HTTP request
			return request.get("prompt"), request.get("max_new_tokens", 256)
	
		def predict(self, x):
			prompt, max_new_tokens = x
			
			# 3. Format inputs for chat-based LLMs
			messages = [{"role": "user", "content": prompt}]
			inputs = self.tokenizer.apply_chat_template(
				messages, 
				add_generation_prompt=True, 
				return_tensors="pt",
				return_dict=True
			).to(self.device)
			
			# 4. Generate text tokens
			outputs = self.model.generate(
				**inputs, 
				max_new_tokens=max_new_tokens,
				eos_token_id=self.tokenizer.eos_token_id
			)
			
			# 5. Extract only the newly generated tokens
			generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
			return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
	
		def encode_response(self, output):
			# 6. Package the generated text into a JSON response
			return {"response": output}
	
	if __name__ == "__main__":
		# Initialize the LitAPI implementation
		api = LLMAPI()
		
		# Configure the production server with dynamic batching
		server = ls.LitServer(api, accelerator="auto")
		
		# Start the server on port 8000
		server.run(port=8000)

	> hf auth login #hugging face to download model
	  python app.py

	http://127.0.0.1:8000/docs	
	input
	{
	  "prompt": "What are the three primary colors?",
	  "max_new_tokens": 128
	}
	
	output
	{
	  "response": "The three primary colors are red, yellow, and blue. These colors cannot be created by mixing other colors together."
	}
	
	curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"prompt": "What are the three primary colors?", "max_new_tokens": 128}'
	
	Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"prompt":"What are the three primary colors?","max_new_tokens":30}' -TimeoutSec 180 | ConvertTo-Json
	{
		"response":  "The three primary colors are red, blue, and yellow. These colors cannot be made by mixing other colors; they are the base colors from which all"
	}
	```
* Retrieval-Augmented Generation (RAG) Code -
	* To deploy a RAG pipeline using LitServe, encapsulate both the vector db retrieval step and the LLM generation step inside the single LitAPI class [LitServe].
	* Because RAG applications perform heavy I/O tasks (searching a database) alongside heavy compute tasks (running the LLM), LitServe’s architecture helps you isolate these steps efficiently.
	* This code implements a complete RAG workflow using ChromaDB (a lightweight vector database) and a Llama 3.2 model via Hugging Face.
	```
	import litserve as ls
	import chromadb, torch
	from transformers import AutoModelForCausalLM, AutoTokenizer
	
	class RAGAPI(ls.LitAPI):
		def setup(self, device):
			# 1. Initialize the vector database client
			# In production, change this to your persistent cloud DB (e.g., Pinecone, Qdrant)
			self.chroma_client = chromadb.Client()
			self.collection = self.chroma_client.get_or_create_collection(name="knowledge_base")
	
			# Seed some mock domain data into the vector DB for demonstration
			self.collection.add(
				documents=[
				"LitServe is an open-source AI engine built on top of FastAPI by Lightning AI.",
				"To optimize performance, LitServe provides built-in dynamic batching.",
				"LitServe supports native streaming using standard Python generators."
				],
				ids=["doc1", "doc2", "doc3"]
			)
	
			# 2. Load the LLM and its Tokenizer into memory
			self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
			self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
			self.model = AutoModelForCausalLM.from_pretrained(
				self.model_name,
				dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
			).to(device)
			self.device = device
	
		def decode_request(self, request):
			# Extract the user's query from the incoming HTTP payload
			return request.get("query")
	
		def predict(self, query):
			# 3. Retrieve relevant context from the vector database
			results = self.collection.query(
				query_texts=[query],
				n_results=2
			)
			# Flatten retrieved document texts into a single context string
			context = " ".join(results['documents'][0])
	
			# 4. Construct a RAG Prompt injecting the context
			rag_prompt = (
				f"Context information is below.\n"
				f"---------------------\n"
				f"{context}\n"
				f"---------------------\n"
				f"Given the context information and not prior knowledge, "
				f"answer the query: {query}"
			)
	
			# 5. Tokenize and pass to the LLM
			messages = [{"role": "user", "content": rag_prompt}]
			inputs = self.tokenizer.apply_chat_template(
				messages, 
				add_generation_prompt=True, 
				return_tensors="pt",
				return_dict=True
			).to(self.device)
	
			outputs = self.model.generate(
				**inputs, 
				max_new_tokens=150,
				eos_token_id=self.tokenizer.eos_token_id
			)
	
			generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
			return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
	
		def encode_response(self, output):
			# Package and return the final answer
			return {"answer": output}
	
	if __name__ == "__main__":
		api = RAGAPI()
		server = ls.LitServer(api, accelerator="auto")
		server.run(port=8000)
	
	{
	  "query": "What optimization feature does LitServe provide?"
	}
	
	{
	  "answer": "LitServe provides built-in dynamic batching to optimize performance."
	}
	```
  
#### [FastMCP](https://gofastmcp.com/getting-started/welcome) created by [prefect](https://www.prefect.io/)
* It's a high-level, Pythonic framework designed to make building MCP servers and clients as effortless as writing standard FastAPI applications.
* it abstracts away the complex JSON-RPC plumbing, protocol handshakes, and manual schema writing required by the raw MCP SDK.
* Core Concepts
	* Servers: Expose local functions or data sources as discoverable MCP tools, resources, and prompts using simple decorators like @mcp.tool.
	* Clients: Provide a typed, programmatic interface (fastmcp.Client) to connect and interact with MCP servers asynchronously without manual connection lifecycle management.
	* Automatic Schemas: Automatically inspects Python type hints and docstrings to generate valid JSON schemas for Large Language Models (LLMs).
	* Transports: Supports standard communication layers like local STDIO and remote HTTP streams seamlessly.
* Code - Setting up a basic server
	```
	from fastmcp import FastMCP
	
	mcp = FastMCP(name="MyServer")
	
	@mcp.tool
	def greet(name: str) -> str:
		"""Greet a user by name."""
		return f"Hello, {name}!"
	
	if __name__ == "__main__":
		mcp.run()
	
	• claude_desktop_config.json
	{
	  "mcpServers": {
		"my-fastmcp-server": {
		  "command": "uv",
		  "args": ["run", "--with", "fastmcp", "path/to/your/server.py"]
		}
	  }
	}
	
	VSC
	{
		"servers": {
			"doc-writer": {
				"url": "http://127.0.0.1:8000/sse",
				"type": "http"
			},
			"mcp-rag-app": {
				"type": "stdio",
				"command": "C:/ai/mcp/.venv/Scripts/python.exe",
				"args": ["C:/ai/mcp/usecase_02/server.py"]
			},
			"15-mcp-apps": {
				"url": "http://localhost:3001/mcp",
				"type": "http"
			}
		},
		"inputs": []
	}
	
	client code
	
	import asyncio
	from fastmcp import Client
	
	async def main() -> None:
		# Connect to a local server script (auto-manages subprocess transport)
		# Or use a URL for HTTP: Client("http://localhost:8000/mcp")
		async with Client("server.py") as client:
			# List available tools
			tools = await client.list_tools()
			print("Available tools:", [t.name for t in tools])
	
			# Call a specific tool
			result = await client.call_tool("greet", {"name": "Alice"})
			print("Tool result:", result.data)
	
	if __name__ == "__main__":
		asyncio.run(main())
	```
#### [Ollama](https://ollama.com/)
* It's a free, open-source platform that lets you run AI LLMs (like Llama, Gemma, DeepSeek, and Mistral)) directly on your own/local computer (CPU or GPU) rather than in the cloud.
* What Ollama Does
	• Complete Privacy: Your prompts, files, and data never leave your machine.
	• No Subscriptions or API Fees: Running local models is completely free after you own the hardware.
	• Offline Capability: You can use AI models without an internet connection.
	• Like Docker for AI: It acts as a lightweight package manager that simplifies downloading, managing, and running complex AI weights with a single command.
* Key Features & How It Works
	* CLI - manage models through your terminal, 
		• ollama run <model>: Downloads and starts a chat session with a model.
		• ollama pull <model>: Downloads a model to your machine without launching it.
		• ollama list (shows all installed models), ollama rm <model> (Deletes a model)
	• Desktop App & GUIs
	• Built-in HTTP API: It exposes a local server (default port 11434), allowing developers to connect Ollama models to custom apps, Python scripts, or tools like LangChain and Open WebUI.
* Hardware Requirements
	* 8 GB RAM: Can run smaller models (roughly 3 to 4 billion parameters).
	* 16 GB RAM: Can handle mid-sized models up to around 20 billion parameters.
	* 32 GB RAM or more: Supports larger models (27 to 30+ billion parameters).
		* Best for General Chat & Logic: Llama 3.3 (70B) or Llama 3.1 (70B) ``` Command: ollama run llama3.3:70b ```
		* Best for Coding, mathematics, multilingual tasks & Speed: Qwen 2.5 (14B or 32B) ```qwen2.5:32b or qwen2.5:14b ```
		* Best for Advanced Reasoning (complex problem-solving, math, and deep logical thinking) : DeepSeek-R1 (14B or 32B) ``` deepseek-r1:32b ```
		* Best "Lightweight & Fast" All-Rounder ( daily text generation, summarizing, and light coding) : Mistral (7B) or Gemma 2 (9B) ``` mistral or gemma2:9b ```

#### [CartesiaAI](https://www.cartesia.ai/)
* Cartesia AI is an AI research lab and voice platform that builds ultra-fast, human-like voice generation and conversational AI models.
* Core Technology & Architecture
	* State Space Models (SSMs): Instead of traditional Transformer architectures, Cartesia builds its models using State Space Models, Mamba, and H-Nets.
	* This allows continuous, real-time processing with low latency, lower costs, and long-context capabilities.
* Key Products & Features
	* Sonic (Text-to-Speech): A real-time voice synthesis engine that delivers hyper-realistic, emotionally expressive speech with sub-90-millisecond latency.
		* Costs roughly 1 credit per character generated. 
	* Ink (Speech-to-Text): A real-time transcription model built for live dialogue, handling background noise, accents, and telephony artifacts.
		* Billed by audio duration depending on the endpoint. For example, ink-2 real-time streaming via WebSockets consumes 3 credits per second (including silence). 
	* Line (Managed Agents Platform): A voice agent development platform that combines Ink, LLMs and Sonic to manage turn-taking, tools, and telephony for automated conversations.
	* Voice Cloning: Allows users to clone voices from short audio clips (around 10 seconds) across dozens of languages and accents.
* Industries: Widely used in customer support, healthcare (patient scheduling and intake), finance, and government sectors.
* Code - pass text input and stream the raw audio bytes directly from the Sonic model:
	```
	pip install cartesia
	
	import os
	from dotenv import load_dotenv
	from cartesia import Cartesia
	load_dotenv()
	
	client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
	
	# Define parameters (Voice IDs can be found in the Cartesia playground)
	voice_id = "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4"
	model_id = "sonic-3.5"
	
	# Request streaming audio bytes
	response = client.tts.generate(
		model_id=model_id,
		transcript="Hello! This is a real-time speech sample generated by Cartesia AI.",
		voice=voice_id,
		output_format={
			"container": "raw",
			"encoding": "pcm_f32le",
			"sample_rate": 44100
		}
	)
	
	response.write_to_file("cartesia-sonic-3.6_test.wav")
	
	# Process or play back the audio stream chunks
	for chunk in response:
		# chunk contains raw audio bytes ready for a playback device or websocket client
		pass
	```

#### [AssemblyAI](https://www.assemblyai.com/)
* It's a leading Speech AI infrastructure platform that provides production-ready AI models to transcribe, understand, and build applications around voice and spoken data.
* Instead of building complex ML models from scratch, developers connect to AssemblyAI via a simple API to process massive amounts of audio and video.
* 🛠️ Core Capabilities
	1. Speech-to-Text (STT) - Pre-recorded Audio (Transcribes files in 99+ languages), Real-time Streaming, Context-Aware Prompting
	2. Audio Intelligence - Sentiment Analysis, PII Redaction, Smmarization & Chapters, Content Moderation
	3. AI Voice Agent Infrastructure - Turn Detection (Knowing exactly when a user stops speaking), Interruption Handling (Instantly silencing the AI if a human cuts in), Tool Calling
* Code
	```
	import assemblyai as aai
	from assemblyai.prerecorded.v2 import Transcriber, TranscriptionConfig
	
	config = TranscriptionConfig(
		speech_models=["universal-3-5-pro"],
		language_detection=True,
		speaker_labels=True,
	)
	
	audio_file = "https://assembly.ai/wildfires.mp3"
	transcriber = Transcriber(api_key="sdfsfsd")
	transcript = transcriber.transcribe(audio_file, config=config)
	
	print(transcript.text)
	for utterance in transcript.utterances:
		print(f"Speaker {utterance.speaker}: {utterance.text}")
	```

#### [LlamaIndex](https://www.llamaindex.ai/) to power RAG
* It's an open-source data and agent framework that connects LLMs to private, domain-specific, or external data sources.
* What LlamaIndex is Used For
	* Context Augmentation & RAG: LLMs are trained on public data and do not know your private company files, databases, or recent documents. LlamaIndex supplies this missing context at query time through Retrieval-Augmented Generation (RAG).
	* Data Ingestion & Connectors: It pulls data from over 100+ sources and formats (such as PDFs, SQL databases, Notion, Slack, and APIs) using LlamaHub.
	* Document Parsing & Extraction: Through tools like LlamaParse, it converts complex unstructured files (like PDFs with tables, charts, or images) into clean formats like Markdown or JSON for AI use.
	* Indexing & Search: It breaks documents into chunks and converts them into vector embeddings or structured indices (like vector stores, list indexes, or knowledge graphs) for fast semantic search.
	* AI Agents & Workflows: It allows developers to build autonomous multi-step agents that can reason, pick tools, query data, and execute tasks.
* Code
	```
	from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
	
	from llama_index.llms.ollama import Ollama
	ollama_llm = Ollama(model="llama3.2:latest")
	
	from llama_index.embeddings.ollama import OllamaEmbedding
	embed_model = OllamaEmbedding(model_name='nomic-embed-text:latest')
	
	Settings.llm = ollama_llm
	Settings.embed_model = embed_model
	
	# 1. Load all documents from a local folder
	documents = SimpleDirectoryReader("C:\\doc\\sequencediagrams").load_data()
	
	# 2. Automatically chunk, embed, and index the text
	index = VectorStoreIndex.from_documents(documents)
	
	# 3. Create a query engine and ask a question
	query_engine = index.as_query_engine()
	response = query_engine.query("summary ca drop")
	print(response)
	```

#### [Livekit](https://livekit.com/) for orchestration
* It's an open-source real-time communication platform designed to make it easy for developers to build live audio, video, and data streaming into their applications.
* Popularity recently as the go-to infrastructure for real-time AI voice and multimodal agents (like ChatGPT’s Advanced Voice Mode).
* It provides a specialized framework (in Python and Node) to connect LLMs, Speech-to-Text (STT), and Text-to-Speech (TTS) models directly into a WebRTC room.
* Key Use Cases
	* Voice & Multimodal AI Agents: Building realistic AI customer service bots, language tutors, or interactive voice assistants that talk, listen, and interrupt just like humans.
	* Video Conferencing & Telehealth: Creating embedded video chat apps similar to Zoom or Google Meet.
	* Live Streaming & Interactive Broadcasting: Real-time streaming for gaming, auctions, or town halls where latency needs to be instant.
	* Robotics & Teleoperation: Streaming video from drones or robots to a browser control interface with negligible delay.
 * Code
	<img alt="image" src="https://github.com/user-attachments/assets/487eafff-eb7c-433c-9a17-e514025a5a0e" />

	```
	LIVEKIT_API_KEY=asdasda
	LIVEKIT_API_SECRET=asdasdas
	
	from dotenv import load_dotenv
	
	from livekit import agents
	from livekit.agents import AgentServer, AgentSession, Agent, inference, room_io, TurnHandlingOptions
	from livekit.plugins import ai_coustics
	
	load_dotenv()
	
	class Assistant(Agent):
		def __init__(self) -> None:
			super().__init__(
				instructions="""You are a helpful voice AI assistant.
				You eagerly assist users with their questions by providing information from your extensive knowledge.
				Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
				You are curious, friendly, and have a sense of humor.""",
			)
	
	server = AgentServer()
	
	@server.rtc_session(agent_name="my-agent")
	async def my_agent(ctx: agents.JobContext):
		session = AgentSession(
			stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
			llm=inference.LLM(model="google/gemma-4-31b-it"),
			tts=inference.TTS(
				model="fishaudio/s2.1-pro",
				voice="fa4c9eb3dccc4806b382b40d61c6b10a",
			),
			turn_handling=TurnHandlingOptions(
				turn_detection=inference.TurnDetector(),
			),
		)
	
		await session.start(
			room=ctx.room,
			agent=Assistant(),
			room_options=room_io.RoomOptions(
				audio_input=room_io.AudioInputOptions(
					noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
				),
			),
		)
	
		await session.generate_reply(
			instructions="Greet the user and offer your assistance."
		)
	
	if __name__ == "__main__":
		agents.cli.run_app(server)
	```
 
	<img alt="image" src="https://github.com/user-attachments/assets/15165c1b-a379-4051-90b2-c8150e4b7492" />
	
	```
	real time
	
	from dotenv import load_dotenv
	
	from livekit import agents
	from livekit.agents import AgentServer, AgentSession, Agent, inference, room_io
	from livekit.plugins import ai_coustics
	
	load_dotenv()
	
	class Assistant(Agent):
		def __init__(self) -> None:
			super().__init__(instructions="You are a helpful voice AI assistant.")
	
	server = AgentServer()
	
	@server.rtc_session(agent_name="my-agent")
	async def my_agent(ctx: agents.JobContext):
		session = AgentSession(
			llm=inference.RealtimeModel(
				model="openai/gpt-realtime-2.1",
				voice="marin",
			)
		)
	
		await session.start(
			room=ctx.room,
			agent=Assistant(),
			room_options=room_io.RoomOptions(
				audio_input=room_io.AudioInputOptions(
					noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
				),
			),
		)
	
		await session.generate_reply(
			instructions="Greet the user and offer your assistance. You should start by speaking in English."
		)
		
	if __name__ == "__main__":
		agents.cli.run_app(server)
	```
