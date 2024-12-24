import google.generativeai as genai
from crewai import Agent, Task, Crew, LLM

gemini_api_key= "YOUR_API_KEY"
genai.configure(api_key=gemini_api_key)

# Initialize the LLM with the Gemini model
llm = LLM(
    api_key=gemini_api_key,
    model='gemini/gemini-1.5-flash'
)
# Define the Content Researcher agent
researcher = Agent(
    role="Content Researcher",
    goal="Research the main themes of {topic}",
    backstory="You're working on researching the key points about {topic}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the Content Writer agent
writer = Agent(
    role="Content Writer",
    goal="Write insightful and accurate summaries about {topic}",
    backstory="You're working on summarizing the key points about {topic}.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define the research task
research_task = Task(
    description="Research the main themes of {topic} and provide bullet points with brief descriptions.",
    agent=researcher,
    expected_output="A list of key themes with descriptions"
)

# Define the writing task
writing_task = Task(
    description="Write a 200-word summary about the topic, incorporating research provided by the Content Researcher.",
    agent=writer,
    expected_output="A 200-word summary of the topic"
)

# Create a Crew and add the agents and tasks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Start the Crew workflow and capture the result
try:
    result = crew.kickoff(inputs={"topic": "Agentic AI"})
    print("Workflow completed successfully:")
    print(result)
except Exception as e:
    print("An error occurred during the workflow:")
    print(str(e))
