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
import sys       # Add this import
from langchain.agents import AgentType, create_json_agent, initialize_agent, load_tools
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

tab1, tab2,tab3,tab4 = st.tabs(
    [ "Chat Bot","Upload PDF Files","Learning Space for Students","Decision Support for Organizations"]
)
with tab3:
    st.markdown("""
<style>
.big-font {
  font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)
    st.markdown('<p class="big-font">This space is under construction</p>', unsafe_allow_html=True)
    st.caption(
               "Dive into cloud development with our agent-powered learning platform! We guide you through a structured curriculum, exploring multiple cloud providers without the pressure of picking one."
      "Build your skills and knowledge in a risk-free environment.")

  from langchain_community.utilities import GoogleSearchAPIWrapper
  from langchain_core.tools import Tool

  search = GoogleSearchAPIWrapper()

  tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
  )
  
  # Simple Curriculum (assuming zero cloud development experience)
    curriculum = {
        1: "What is Cloud Computing?",
        2: "What are the different types of Cloud Services (IaaS, PaaS, SaaS)?",
        3: "Who are the major Cloud Providers (AWS, Azure, GCP)?",
        4: "What are the benefits of using Cloud Computing?",
        5: "Can you explain some common Cloud Computing use cases?"
    }

    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    # Display current curriculum step
    st.write(f"**Curriculum Step {st.session_state.current_step}:** {curriculum[st.session_state.current_step]}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if question := st.chat_input("Ask your question here:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        # Load the model
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1, api_key=google_api_key)

        # Initialize the agent with tools
        # tools = load_tools(["python_repl"], llm=model)
        agent = initialize_agent(tool, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

        # Get the response
        response = agent.run(question)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)

        # Move to the next step in the curriculum
        if st.session_state.current_step < len(curriculum):
            st.session_state.current_step += 1
        else:
            st.write("Congratulations! You have completed the curriculum.")
    # --- CHANGES END HERE ---


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
