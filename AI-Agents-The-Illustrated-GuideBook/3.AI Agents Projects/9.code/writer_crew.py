from pydantic import BaseModel
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
if __package__:
    from .custom_tools import FirecrawlWebSearchTool
else:
    from custom_tools import FirecrawlWebSearchTool

llm = LLM(model="ollama/llama3.2:3b")

class Chapter(BaseModel):
    """Chapter of the book"""

    title: str
    content: str

@CrewBase
class ChapterWriterCrew:
    """Chapter Writer Crew"""

    agents_config = "writer_agents.yaml"
    tasks_config = "writer_tasks.yaml"

    @agent
    def topic_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_researcher"],
            tools=[FirecrawlWebSearchTool()],
            llm=llm,
        )

    @task
    def research_topic(self) -> Task:
        return Task(config=self.tasks_config["research_topic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"], llm=llm)

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )