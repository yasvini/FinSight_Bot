import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API key is available
if not GOOGLE_API_KEY:
    st.error("❌ Missing API Key! Please check your .env file.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Set API Key

# Streamlit UI
st.set_page_config(page_title="FinSightBot: News Research Tool", page_icon="📰", layout="wide")
st.title("📰 FinsightBot: News Research Tool")
st.sidebar.title("Navigation")

# Sidebar for URL input and tabs for better organization
with st.sidebar:
    st.header("Input URLs")
    with st.form("url_form"):
        urls = [st.text_input(f"Enter URL {i + 1}", placeholder=f"Paste URL {i + 1} here") for i in range(3)]
        process_url_clicked = st.form_submit_button("🔄 Process URLs")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Enter up to 3 valid URLs and process them to build the database.")

# Main tabs for improved UI organization
tab1, tab2 = st.tabs(["🔗 Data Processing", "💬 Ask Questions"])

# FAISS index folder path
faiss_index_folder = "faiss_index"

# Placeholder for status updates
main_placeholder = st.empty()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9)

# Tab 1: Data Processing
with tab1:
    st.header("🔗 Data Processing")
    if process_url_clicked:
        with st.spinner("🔄 Loading and processing data..."):
            try:
                # Remove empty URLs and load data
                valid_urls = [url for url in urls if url]
                if not valid_urls:
                    st.warning("⚠️ Please enter at least one valid URL.")
                else:
                    loader = UnstructuredURLLoader(urls=valid_urls)
                    data = loader.load()

                    # Split text into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    docs = text_splitter.split_documents(data)

                    # Create embeddings & FAISS index
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore_gemini = FAISS.from_documents(docs, embeddings)

                    # Save FAISS index
                    vectorstore_gemini.save_local(faiss_index_folder)

                    st.success("✅ Data processing complete! Database is ready for queries.")

            except Exception as e:
                st.error(f"❌ Error processing URLs: {e}")

# Tab 2: Query Section
with tab2:
    st.header("💬 Ask Questions")
    query = st.text_input("🔍 Ask your question here:", placeholder="Type your query...")
    
    if query:
        if os.path.exists(faiss_index_folder):
            try:
                # Define embeddings before loading FAISS
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                # Load FAISS index with safe deserialization
                vectorstore = FAISS.load_local(faiss_index_folder, embeddings, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever()

                # Use RetrievalQA with chain_type="stuff"
                chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

                # Run query
                with st.spinner("🔍 Searching for answers..."):
                    result = chain.invoke({"query": query})
                    answer = result["result"] if isinstance(result, dict) else result

                st.subheader(f"🔎 Query: {query}")
                st.markdown(f"**Answer:**\n{answer}")

            except Exception as e:
                st.error(f"❌ Error retrieving answer: {e}")
        else:
            st.warning("⚠️ Please process URLs first to build the database.")

# Footer section
st.sidebar.markdown("---")
st.sidebar.markdown("🎯 **Developed by RockyBot**")
st.sidebar.markdown("🌟 For any issues or feedback, feel free to contact us!")
