import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io
from langchain_community.vectorstores import Chroma
import pysqlite3  # Add this import
import sys      # Add this import
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.utilities import GoogleSearchAPIWrapper
import pandas as pd
import re
import plotly.express as px
from graphviz import Source
import graphviz

# Function to validate email format
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

# Swap sqlite3 with pysqlite3-binary
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

st.title("Cloud Current")
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Developed by Varun Sankuri</p>', unsafe_allow_html=True)
st.caption("Frustrated with ChatGPT and Google Gemini giving you outdated cloud info? Big box models can't keep up with the Cloud's rapid pace."
            " CloudCurrent is updated much more frequently and also lets you upload your OWN PDFs to get the most accurate, "
            "'up-to-the-minute answers'. Try it now!")
# st.caption("Example questions: Compare S3 storage classes and their use cases, or upload a file and ask the bot to Summarize the file")
st.caption("For questions contact cloudcurrentapp@gmail.com")

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
# google_api_key = os.getenv("GOOGLE_API_KEY")
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Chat Bot", "Upload PDF Files", "Decision Support for Organizations","Learning Space for Students"]
)


with tab4:
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.caption(
            "Dive into cloud development with our learning platform! We guide you through a structured curriculum, exploring multiple cloud providers."
            "Build your skills and knowledge in a risk-free environment.")

    # Simple Curriculum
    curriculum = {
    
    1: {
        "question": """<h3>Explain the three main types of Cloud Services:</h3>

<ul>
  <li><b>IaaS (Infrastructure as a Service)</b></li>
  <li><b>PaaS (Platform as a Service)</b></li> 
  <li><b>SaaS (Software as a Service)</b></li> 
</ul>

<p>Provide real-world examples of each.</p>""",
        "hints": [
            "<p><b>IaaS:</b> What aspects of the infrastructure do you manage?</p>",
            "<p><b>PaaS:</b> What tools and resources are provided for developers?</p>",
            "<p><b>SaaS:</b> What kind of applications are typically delivered as SaaS?</p>"
        ]
    },
        2: {
        "question": """<h3>What is Cloud Computing in your own words?</h3> 

<p>How does it differ from traditional on-premises infrastructure?</p>""",
        "hints": [
            "<p>Think about where the hardware and software are located.</p>",
            "<p>Consider the concepts of on-demand access and scalability.</p>",
            "<p>How do you pay for cloud services versus owning your own hardware?</p>"
        ]
    },
    3: {
        "question": """<h3>Compare and contrast the three major Cloud Providers:</h3>

<ul>
  <li><b>AWS (Amazon Web Services)</b></li>
  <li><b>Azure (Microsoft Azure)</b></li>
  <li><b>GCP (Google Cloud Platform)</b></li>
</ul>

<p>What are their strengths and weaknesses?</p>""",
        "hints": [
            "<p>Consider factors like market share, global reach, and pricing models.</p>",
            "<p>What specific services are each provider known for?</p>",
            "<p>Are there any industry-specific offerings or certifications?</p>"
        ]
    },
    4: {
        "question": """<h3>Imagine you're a consultant advising a company on migrating to the cloud.</h3> 

<p>What benefits would you highlight to convince them?</p>""",
        "hints": [
            "<p>Think about cost savings, scalability, and increased efficiency.</p>",
            "<p>How does cloud computing improve security and disaster recovery?</p>",
            "<p>What about innovation and access to new technologies?</p>"
        ]
    },
    5: {
        "question": """<h3>Explore some common Cloud Computing use cases across different industries.</h3> 

<p>How is the cloud transforming businesses?</p>""",
        "hints": [
            "<p>Consider examples in healthcare, finance, e-commerce, and media.</p>",
            "<p>How is the cloud used for data storage, analytics, and AI?</p>",
            "<p>What about mobile app development, IoT, and gaming?</p>"
        ]
    },
    6: {
        "question": """<h3>What are some of the challenges and risks associated with Cloud Computing?</h3> 

<p>How can these be mitigated?</p>""",
        "hints": [
            "<p>Think about security breaches, vendor lock-in, and compliance issues.</p>",
            "<p>What about data privacy, outages, and unexpected costs?</p>",
            "<p>How can companies ensure business continuity in the cloud?</p>"
        ]
    },
    7: {
        "question": """<h3>Discuss the future of Cloud Computing.</h3> 

<p>What emerging trends and technologies will shape its evolution?</p>""",
        "hints": [
            "<p>Consider serverless computing, edge computing, and cloud-native development.</p>",
            "<p>What about the role of AI, machine learning, and quantum computing in the cloud?</p>",
            "<p>How will cloud computing impact sustainability and environmental concerns?</p>"
        ]
    }
}
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    # Display current curriculum step
    st.write(f"**Curriculum Step {st.session_state.current_step}:**", unsafe_allow_html=True) 
    st.markdown(curriculum[st.session_state.current_step]["question"], unsafe_allow_html=True)

    # Display hints (using st.expander for better organization)
    with st.expander("Hints"): 
        for hint in curriculum[st.session_state.current_step]["hints"]:
            st.markdown(hint, unsafe_allow_html=True)

       # Initialize chat history (specific to Tab 3)
    if "messages_tab3" not in st.session_state:  # <-- Changed variable name
        st.session_state.messages_tab3 = []  # <-- Changed variable name

    # Display chat messages from history (specific to Tab 3)
    for message in st.session_state.messages_tab3:  # <-- Changed variable name
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if question := st.chat_input("Answer the question here to the best of your knowledge:"):
        # ... (add user message to chat history and display it)

        # Load the LLM
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, api_key=google_api_key)

        # Construct the prompt with the curriculum step, user question, and evaluation instructions
        prompt = f"""You are a helpful AI assistant helping train people on Cloud development and deployment. 

        Curriculum Step {st.session_state.current_step}: {curriculum[st.session_state.current_step]["question"]}

        User Answer: {question}

        Evaluate the user's answer to the curriculum question. 
        The answer should be comprehensive and accurate.
        Provide feedback to the user, including whether the answer is correct or needs improvement.
        If the answer needs improvement, provide specific guidance on what to improve. 

        Evaluation and Feedback:""" 

        # Get the response
        response = model.predict(prompt)

        # Add assistant response to chat history
        st.session_state.messages_tab3.append({"role": "user", "content": question})  # <-- Changed variable name
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)

        # Check if the LLM deems the answer sufficient
        # (This is a simplified check, you might need more sophisticated evaluation logic)
        # Check if the LLM deems the answer sufficient
        if "correct" in response.lower():
            st.success("Correct! Moving on to the next question.")

            # Increment the current step *before* displaying the next question
            st.session_state.current_step += 1

            # Check if there are more questions
            if st.session_state.current_step <= len(curriculum):
                st.write(f"**Curriculum Step {st.session_state.current_step}:**", unsafe_allow_html=True)
                st.markdown(curriculum[st.session_state.current_step]["question"], unsafe_allow_html=True)
                # Re-display hints for the new question
                with st.expander("Hints"):
                    for hint in curriculum[st.session_state.current_step]["hints"]:
                        st.markdown(hint, unsafe_allow_html=True)


            else:
                st.write("Congratulations! You have completed the curriculum.")
        else:
            st.error("Incorrect. Please try again.")


with tab3:
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50; /* Green */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #3e8e41; /* Darker green */
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Decision Support for Organizations</p>', unsafe_allow_html=True)
    st.caption(
        "Overwhelmed by complex business decisions? Cloud Current simplifies cost analysis, architecture design, and more. "
        "Make data-driven choices with confidence using our intuitive tools and visualizations. "
        "Try it for free and see the difference Cloud Current can make in your business."
    )

    st.subheader("Cost Analysis")

    # --- Cost Analysis Input ---
    st.markdown("**Estimate your cloud costs:**")
    col1, col2, col3, col4 = st.columns(4)
    with col2:
        compute_hours = st.number_input("Compute Hours/Month", min_value=0, value=100)
        storage_gb = st.number_input("Storage (GB)/Month", min_value=0, value=50)
    with col3:
        bandwidth_tb = st.number_input("Bandwidth (TB)/Month", min_value=0.0, value=1.0)
        region = st.selectbox("Region", ["US East", "US West", "Europe", "Asia"])
    with col4:
        os = st.selectbox("Operating System", ["Linux", "Windows"])
        compute_service = st.selectbox("Compute Service", ["Basic Compute", "High-Performance Compute"])
    with col1:
        service_provider = st.selectbox("Service Provider", ["AWS", "Azure", "Google Cloud"])

    # --- Sample Cost Data (Replace with actual cloud pricing data) ---
    cost_data = {
    "Service Provider": ["AWS", "Azure", "Google Cloud", "AWS", "Azure", "Google Cloud", "AWS", "Azure", "Google Cloud", "AWS", "Azure", "Google Cloud"],
    "Region": ["US East", "US East", "US East", "US West", "US West", "US West", "Europe", "Europe", "Europe", "Asia", "Asia", "Asia"],
    "Basic Compute": [0.10, 0.12, 0.09, 0.12, 0.13, 0.10, 0.14, 0.15, 0.13, 0.16, 0.17, 0.15],
    "High-Performance Compute": [0.20, 0.22, 0.19, 0.22, 0.23, 0.20, 0.24, 0.25, 0.23, 0.26, 0.27, 0.25],
    "Storage": [0.05, 0.06, 0.04, 0.06, 0.07, 0.05, 0.07, 0.08, 0.06, 0.08, 0.09, 0.07],
    "Bandwidth": [0.09, 0.10, 0.08, 0.10, 0.11, 0.09, 0.11, 0.12, 0.10, 0.12, 0.13, 0.11],
    "Linux": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Windows": [0.02, 0.025, 0.015, 0.02, 0.025, 0.015, 0.022, 0.027, 0.017, 0.025, 0.03, 0.02],
    }

    cost_df = pd.DataFrame(cost_data)

    # --- Cost Calculation ---
    def calculate_cost(compute_hours, storage_gb, bandwidth_tb, region, os, compute_service, service_provider):
        try:
            filtered_df = cost_df[
                (cost_df["Region"] == region) & (cost_df["Service Provider"] == service_provider)
            ]
            compute_cost = compute_hours * filtered_df[compute_service].iloc[0]
            storage_cost = storage_gb * filtered_df["Storage"].iloc[0]
            bandwidth_cost = bandwidth_tb * filtered_df["Bandwidth"].iloc[0]
            os_cost = compute_hours * filtered_df[os].iloc[0]
            total_cost = compute_cost + storage_cost + bandwidth_cost + os_cost
            return total_cost
        except (KeyError, IndexError):
            st.error(
                "Error calculating cost. Please check the selected region, service type, and service provider."
            )
            return 0

    # --- Cost Visualization ---
    if st.button("Calculate Cost"):
        total_cost = calculate_cost(
            compute_hours, storage_gb, bandwidth_tb, region, os, compute_service, service_provider
        )

        st.markdown(f"**Estimated Monthly Cost:  $ {total_cost:.2f}**")

        # Sample cost breakdown data for visualization
        cost_breakdown = {
            "Category": ["Compute", "Storage", "Bandwidth", "OS"],
            "Cost": [
                compute_hours
                * cost_df[
                    (cost_df["Region"] == region) & (cost_df["Service Provider"] == service_provider)
                ][compute_service].iloc[0],
                storage_gb
                * cost_df[
                    (cost_df["Region"] == region) & (cost_df["Service Provider"] == service_provider)
                ]["Storage"].iloc[0],
                bandwidth_tb
                * cost_df[
                    (cost_df["Region"] == region) & (cost_df["Service Provider"] == service_provider)
                ]["Bandwidth"].iloc[0],
                compute_hours
                * cost_df[
                    (cost_df["Region"] == region) & (cost_df["Service Provider"] == service_provider)
                ][os].iloc[0],
            ],
        }
        cost_breakdown_df = pd.DataFrame(cost_breakdown)

        fig = px.bar(
            cost_breakdown_df,
            x="Category",
            y="Cost",
            title="Cost Breakdown",
            color="Category",
        )
        st.plotly_chart(fig)

    st.subheader("Architecture Design")
    st.markdown(
        "**Explore common cloud architecture patterns:**"
    )

    arch_patterns = {
        "Web Application (Three-Tier)": {
            "description": """
                A classic three-tier architecture commonly used for web applications. It separates the application into three logical layers: presentation (web servers), application logic (application servers), and data storage (database).
                This pattern offers good scalability, maintainability, and fault isolation.
            """,
            "components": {
                "Load Balancer": "Distributes incoming traffic across multiple web servers, ensuring high availability and responsiveness.",
                "Web Servers (Presentation Tier)": "Handle user interface and interactions, typically serving static content and routing requests to application servers.",
                "Application Servers (Application Tier)": "Process business logic, handle dynamic content generation, and interact with the database.",
                "Database (Data Tier)": "Stores and manages the application's data. Can be relational (e.g., MySQL, PostgreSQL) or NoSQL (e.g., MongoDB, Cassandra).",
                "Caching Layer (Optional)": "Improves performance by storing frequently accessed data in a fast cache (e.g., Redis, Memcached).",
                "CDN (Optional)": "Content Delivery Network distributes static content closer to users, reducing latency.",
            },
            "diagram": """
                graph TD;
                A[Load Balancer] --> B(Web Server 1);
                A --> C(Web Server 2);
                B --> E(Application Server 1);
                C --> F(Application Server 2);
                E --> D{Database};
                F --> D;
                A -.-> G([CDN]);
                B -.-> H([Caching Layer]);
                C -.-> H;
            """,
            "considerations": """
                - **Scalability:** Each tier can be scaled independently based on demand.
                - **Security:** Implement security measures at each tier (e.g., firewalls, access controls).
                - **Database Choice:** Select a database that meets your application's requirements for data consistency, availability, and scalability.
                - **Caching Strategy:** Carefully design your caching strategy to balance performance gains with data consistency.
            """,
        },
        "Microservices": {
            "description": """
                An architectural style that structures an application as a collection of loosely coupled, independently deployable services. Each service focuses on a specific business capability and communicates with other services through lightweight protocols (e.g., REST APIs, gRPC).
                Microservices enable faster development cycles, improved scalability, and greater resilience.
            """,
            "components": {
                "API Gateway": "Provides a single entry point for external clients to access the microservices. Handles routing, authentication, and rate limiting.",
                "Service A, B, C, etc.": "Individual microservices responsible for specific business functions. Each service has its own database and can be developed and deployed independently.",
                "Service Registry/Discovery": "Keeps track of available service instances and their locations, allowing services to discover and communicate with each other dynamically (e.g., Eureka, Consul).",
                "Message Broker (Optional)": "Facilitates asynchronous communication between services using a message queue (e.g., Kafka, RabbitMQ).",
                "Configuration Server (Optional)": "Centralized management of configuration settings for all services.",
            },
            "diagram": """
                graph TD;
                A[API Gateway] --> B(Service A);
                A --> C(Service B);
                A --> D(Service C);
                B --> E{Service Registry};
                C --> E;
                D --> E;
                B -.-> F([Message Broker]);
                C -.-> F;
                B -.-> G([Configuration Server]);
                C -.-> G;
                D -.-> G;
            """,
            "considerations": """
                - **Service Decomposition:** Carefully decompose your application into well-defined, independent services.
                - **Communication:** Choose appropriate inter-service communication mechanisms (synchronous or asynchronous).
                - **Data Management:** Each service manages its own data, requiring careful consideration of data consistency and transactions.
                - **Monitoring and Logging:** Implement comprehensive monitoring and logging to track service health and performance.
                - **Deployment:** Automate the deployment pipeline for each service.
            """,
        },
        "Serverless": {
            "description": """
                A cloud-native development model that allows you to build and run applications without managing servers. The cloud provider automatically provisions and scales the underlying infrastructure based on demand.
                Serverless architectures are event-driven and highly scalable, offering cost savings for applications with variable workloads.
            """,
            "components": {
                "API Gateway": "Handles API requests and routes them to the appropriate Lambda functions.",
                "Lambda Functions (or equivalent)": "Small, stateless functions that execute in response to events (e.g., HTTP requests, messages, database updates).",
                "DynamoDB (or other managed database)": "A fully managed NoSQL database service that scales automatically.",
                "S3 (or other object storage)": "Stores static assets, such as images, videos, and documents.",
                "Event Source": "Triggers the execution of Lambda functions (e.g., S3 events, DynamoDB streams, message queues).",
            },
            "diagram": """
                graph TD;
                A[API Gateway] --> B(Lambda Function 1);
                A --> C(Lambda Function 2);
                B --> D{DynamoDB};
                C --> E{S3};
                F([Event Source]) --> B;
            """,
            "considerations": """
                - **Statelessness:** Lambda functions are stateless, so you need to manage state externally (e.g., in a database or cache).
                - **Cold Starts:** Lambda functions can experience cold starts, which can add latency to the initial invocation.
                - **Vendor Lock-in:** Serverless architectures can lead to vendor lock-in, as they are often tied to a specific cloud provider's services.
                - **Debugging and Monitoring:** Debugging and monitoring serverless applications can be more complex than traditional applications.
            """,
        },
        "Event-Driven Architecture": {
            "description": """
                A software architecture paradigm centered around the production, detection, consumption of, and reaction to events. An event represents a significant change in state. This pattern decouples services, making them more independent, scalable, and resilient.
            """,
            "components": {
                "Event Producer": "Creates and publishes events to an event bus or message broker.",
                "Event Bus/Message Broker": "Receives events from producers and routes them to appropriate consumers (e.g., Kafka, RabbitMQ, AWS SQS/SNS).",
                "Event Consumer": "Subscribes to specific event types and processes them.",
                "Event Store (Optional)": "Stores a log of all events for auditing, debugging, or replaying events.",
            },
            "diagram": """
                graph TD;
                A[Event Producer 1] --> B((Event Bus/Message Broker));
                C[Event Producer 2] --> B;
                B --> D(Event Consumer 1);
                B --> E(Event Consumer 2);
                B --> F{Event Store};
            """,
            "considerations": """
                - **Event Design:** Carefully design your events to be granular, self-contained, and meaningful.
                - **Asynchronous Communication:** Event-driven architectures rely on asynchronous communication, which can introduce complexities in terms of error handling and data consistency.
                - **Eventual Consistency:** Data consistency may be eventual, as services process events independently and at their own pace.
                - **Ordering and Duplication:** Handle event ordering and potential duplication of events, if necessary.
            """,
        },
        "Big Data Architecture": {
            "description": """
                Designed to handle the ingestion, processing, storage, and analysis of massive volumes of data that are too large or complex for traditional database systems. These architectures often involve distributed computing, parallel processing, and specialized tools for big data analytics.
            """,
            "components": {
                "Data Sources": "Various sources of big data, such as IoT devices, social media feeds, log files, and databases.",
                "Data Ingestion": "Tools and processes for collecting and importing data from various sources into the big data system (e.g., Kafka, Flume, Sqoop).",
                "Data Storage": "Distributed file systems (e.g., HDFS) or cloud-based object storage (e.g., S3, Azure Blob Storage) for storing large datasets.",
                "Data Processing": "Frameworks for distributed data processing and analysis (e.g., Hadoop MapReduce, Spark, Flink).",
                "Data Serving/Querying": "Databases or query engines optimized for analytical queries on large datasets (e.g., Hive, Presto, Impala).",
                "Data Visualization/Reporting": "Tools for creating dashboards, reports, and visualizations to analyze the processed data (e.g., Tableau, Power BI).",
            },
            "diagram": """
                graph TD;
                A[Data Sources] --> B(Data Ingestion);
                B --> C{Data Storage};
                C --> D[Data Processing];
                D --> E(Data Serving/Querying);
                E --> F[Data Visualization/Reporting];
            """,
            "considerations": """
                - **Data Volume and Velocity:** Design your architecture to handle the expected volume and velocity of data.
                - **Data Variety:** Consider the different types of data (structured, semi-structured, unstructured) that your system needs to process.
                - **Scalability and Elasticity:** Choose technologies that can scale horizontally to accommodate growing data volumes and processing needs.
                - **Data Security and Governance:** Implement appropriate security measures and data governance policies to protect sensitive data.
            """,
        },
    }

    selected_pattern = st.selectbox("Select Architecture Pattern", list(arch_patterns.keys()))

    st.markdown(arch_patterns[selected_pattern]["description"])
    st.markdown("**Components:**")
    for component, description in arch_patterns[selected_pattern]["components"].items():
        st.markdown(f"- **{component}:** {description}")

     # Render the Graphviz diagram using st.graphviz_chart
    # st.markdown("**Architecture Diagram:**")

    # Create a Graphviz graph directly
    graph = graphviz.Digraph()
    graph.body.extend(arch_patterns[selected_pattern]["diagram"].splitlines())  # Add the diagram definition

    st.graphviz_chart(graph)

    st.markdown("**Key Considerations:**")
    st.markdown(arch_patterns[selected_pattern]["considerations"])

    # --- Contact Form ---
    st.subheader("Contact Us for a Free Trial")
    st.markdown(
        "Interested in learning more about how Cloud Current can help your business? Fill out the form below, and we'll get in touch!"
    )

    # Initialize session state for form submission if it doesn't exist
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        company = st.text_input("Company Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number (Optional)")
        message = st.text_area("Message (Optional)")

        # Submit button for the form
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if not name or not company or not email:
                st.warning("Please fill in all required fields.")
            elif not is_valid_email(email):
                st.warning("Please enter a valid email address.")
            else:
                # Here you would typically send an email or store the form data
                # For this example, we'll just display a success message
                st.session_state.form_submitted = True

    # Display a success message if the form was submitted
    if st.session_state.form_submitted:
        st.success("Thank you for your interest! We'll be in touch soon.")

with tab2:
    st.caption("Although not necessary, you can upload your PDFs here to get more accurate answers/code")
    # File Upload with multiple file selection
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        st.text("PDF Files Uploaded Successfully!")

        # Combine all PDF content
        all_texts = []
        for uploaded_file in uploaded_files:
            # PDF Processing
            pdf_data = uploaded_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            pdf_pages = pdf_reader.pages

            # Extract text from all pages and add to the combined context
            context = "\n\n".join(page.extract_text() for page in pdf_pages)
            all_texts.append(context)

        # Combine all contexts into a single string
        combined_context = "\n\n".join(all_texts)

        # Split Texts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(combined_context)

        # Chroma Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

with tab1:
     # --- Example Questions ---
    st.markdown("**Example Questions:**")
    st.markdown("""
    1.  What are the key differences between AWS Lambda, Azure Functions, and Google Cloud Functions, and when should I choose one over the others for a serverless project?
    2.  I need to design a highly available and scalable web application architecture using GCP. Can you suggest a suitable architecture diagram and explain the role of each component, including load balancing, auto-scaling, and database choices?
    """)
    # --- CUSTOM CSS FOR CHATBOX ---
    st.markdown(
        """
        <style>
        /* Increase height of chat input box */
        div[data-baseweb="textarea"] {
            height: 100px !important; /* Adjust height as needed */
        }
        
        /* Increase height of chat message containers */
        .stChatMessage {
            height: 200px; /* Adjust if you want a fixed height, otherwise 'auto' is good */
            min-height: 100px; /* Minimum height to prevent very small boxes */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

   

    # Get user input
    if question := st.chat_input("Ask your Cloud related questions here."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Get Relevant Documents (only if files were uploaded)
        if uploaded_files:
            docs = vector_index.get_relevant_documents(question)
        else:
            docs = []  # No documents to provide

        # Define Prompt Template
        prompt_template = """
        You are a helpful AI assistant helping people answer their Cloud development and
        deployment questions. Answer the question as detailed as possible from the provided context,
        make sure to provide all the details and code if possible, if the answer is not in
        provided context use your  knowledge or imagine an answer but never say that you don't have an answer
        or can't provide an answer based on current context ",

        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

        # Create Prompt
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        # Load QA Chain
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1, api_key=google_api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        # Get Response
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['output_text']})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response['output_text'])
