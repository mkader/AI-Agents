import sys, datetime, os
import streamlit as st
from crewai import Crew, Process, Task, Agent, LLM
from browserbase import browserbase
from kayak import kayak_hotels
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="🏨 HotelFinder Pro", layout="wide")

# Title and subtitle with custom HTML for blue color
st.markdown("<h1 style='color: #0066cc;'>🏨 HotelFinder Pro</h1>", unsafe_allow_html=True)
st.subheader("Powered by Browserbase and CrewAI")

@st.cache_resource(show_spinner=False)
def load_llm():
    return LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")

# Load environment variables
load_dotenv()  # take environment variables from .env.

# Main content
st.markdown("---")

# Hotel search form
st.header("Search for Hotels")
col1, col2 = st.columns(2)

with col1:
    location = st.text_input("Location", "Enter city, area, or landmark")
    num_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)

with col2:
    check_in_date = st.date_input("Check-in Date", datetime.date.today())
    check_out_date = st.date_input("Check-out Date", datetime.date.today() + datetime.timedelta(days=1))
    # Add more options if needed

search_button = st.button("Search Hotels")

# Initialize agents
hotels_agent = Agent(
    role="Hotels",
    goal="Search hotels",
    backstory="I am an agent that can search for hotels and find the best accommodations.",
    tools=[kayak_hotels, browserbase],
    allow_delegation=False,
    llm=load_llm(),
)

summarize_agent = Agent(
    role="Summarize",
    goal="Summarize hotel information",
    backstory="I am an agent that can summarize hotel details and amenities.",
    allow_delegation=False,
    llm=load_llm(),
)

output_search_example = """
Here are our top 5 hotels in New York for September 21-22, 2024:
1. Hilton Times Square:
   - Rating: 4.5/5
   - Price: $299/night
   - Location: Times Square
   - Amenities: Pool, Spa, Restaurant
   - Booking: https://www.kayak.com/hotels/hilton-times-square
"""

search_task = Task(
    description=(
        "Search hotels according to criteria {request}. Current year: {current_year}"
    ),
    expected_output=output_search_example,
    agent=hotels_agent,
)

output_providers_example = """
Detailed information for hotels in New York (September 21-22, 2024):
1. Hilton Times Square:
   - Room Types: Deluxe King, Double Queen
   - Price Range: $299-$499/night
   - Special Offers: Free breakfast, Free cancellation
   - Booking Options:
     * Kayak: $299/night
     * Hotels.com: $315/night
     * Direct: $325/night
"""

search_booking_providers_task = Task(
    description="Load hotel details and find available booking providers with their rates",
    expected_output=output_providers_example,
    agent=hotels_agent,
)

# Search functionality
if search_button:
    if check_out_date <= check_in_date:
        st.error("Check-out date must be after check-in date!")
    else:
        with st.spinner("Searching for hotels... This may take a few minutes."):
            # Format the request
            request = f"hotels in {location} from {check_in_date.isoformat()} to {check_out_date.isoformat()} for {num_adults} adults"
            
            crew = Crew(
                agents=[hotels_agent, summarize_agent],
                tasks=[search_task, search_booking_providers_task],
                # Avoid [INFO]: Max RPM reached, waiting for next minute to start.
                # limiting it to one request per minute. Tasks may need multiple LLM calls, especially when using tools, so CrewAI pauses once that limit is reached.
                # For local Ollama, remove max_rpm=1 to disable CrewAI’s explicit throttle, or increase it, for example: max_rpm=30,
                #max_rpm=1,
                verbose=True,
                planning=True,
                planning_llm=load_llm(),
            )
            
            # Execute the search
            try:
                result = crew.kickoff(
                    inputs={
                        "request": request,
                        "current_year": datetime.date.today().year,
                    }
                )
                
                # Display results
                st.success("Search completed!")
                st.markdown("## Hotel Results")
                st.markdown(result.raw)
            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")

# Add some information about the app
st.markdown("---")
st.markdown("""
### About HotelFinder Pro
This application uses AI agents to search for hotels and find the best accommodations for you.
Simply enter your desired location, dates, and number of guests to get started.

Features:
- Real-time hotel availability
- Comprehensive price comparison
- Detailed hotel information and amenities
- Multiple booking options
""")