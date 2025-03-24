# FinSight Bot
## Overview

FinSight Bot is an AI-powered financial and stock market news analysis tool designed to retrieve insights from article URLs. It leverages LangChain, Google Gemini API, and FAISS for efficient retrieval-augmented generation (RAG). Users can input URLs, process data, and interact with an LLM to get context-aware responses with source references.

## Features

- Extracts and processes financial and stock market news from URLs.

- Utilizes LangChain's UnstructuredURLLoader for automated web scraping.

- Enhances document processing with RecursiveCharacterTextSplitter for improved text chunking and retrieval efficiency.

- Stores and retrieves insights efficiently using FAISS indexing.

- Provides a interface for user interaction.

- Enables querying with LLM-based responses, including source URL references.

## Tech Stack

- Frontend: Streamlit (or Flask if applicable)

- Backend: Python, LangChain

- Database: FAISS for vector storage

- NLP: Google Gemini API

## Installation

## Prerequisites

- Python 3.8+

- API key for Google Gemini

- Required Python libraries

## Steps

- 1. Clone the repository:
  ```bash
    git clone https://github.com/yasvini/finsight-bot.git
    cd finsight-bot

- 2. Install dependencies:
  ```bash
  pip install -r requirements.txt

- 3. Set up API keys:
  - Create a .env file and add your API key:
    ```bash
    GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key

- 4. Run the application:
  ```bash
  python app.py  # or streamlit run app.py if using Streamlit

## Usage

- Open the web app and enter the article URL.

- The bot will extract and embed content for indexing.

- Users can query financial insights related to the article.

- The response includes key insights along with source references.

## Future Enhancements
- Improve UI/UX for better usability.

- Expand retrieval system with more financial document sources.

- Integrate more advanced AI models for better insights.

## Contributing
  Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
