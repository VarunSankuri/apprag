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
st.caption("Example questions: Compare S3 storage classes and their use cases, or upload a file and ask the bot to Summarize the file")
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
    ["Chat Bot", "Upload PDF Files", "Learning Space for Students", "Decision Support for Organizations"]
)


with tab3:
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
        "question": """<h3>What is Cloud Computing in your own words?</h3> 

<p>How does it differ from traditional on-premises infrastructure?</p>""",
        "hints": [
            "<p>Think about where the hardware and software are located.</p>",
            "<p>Consider the concepts of on-demand access and scalability.</p>",
            "<p>How do you pay for cloud services versus owning your own hardware?</p>"
        ]
    },
    2: {
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

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
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
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)

        # Check if the LLM deems the answer sufficient
        # (This is a simplified check, you might need more sophisticated evaluation logic)
        if "correct" in response.lower(): 
            st.success("Correct! Moving on to the next question.") 

            # Increment the current step *before* displaying the next question 
            st.session_state.current_step += 1  # <-- Moved this line up

            # Check if there are more questions
            if st.session_state.current_step <= len(curriculum):  
                st.write(f"**Curriculum Step {st.session_state.current_step}:**", unsafe_allow_html=True)
                st.markdown(curriculum[st.session_state.current_step]["question"], unsafe_allow_html=True) 
            else:
                st.write("Congratulations! You have completed the curriculum.")
        else:
            st.error("Incorrect. Please try again.") 

with tab4:
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">This space is under construction</p>', unsafe_allow_html=True)
    st.caption(
            "Overwhelmed by complex business decisions? Cloud Current simplifies cost analysis, architecture design, and more."
            "Make data-driven choices with confidence using our intuitive tools and visualizations.  Try it for free and see the difference Cloud Current can make in your business.")

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
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if question := st.chat_input("Ask your Cloud related questions here. For e.g. Compare AWS S3 storage classes and their use cases"):
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
