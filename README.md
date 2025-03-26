# Docu_Mind-chatbot
DocuMind DocuMind is an AI-powered chatbot that transforms your PDFs into a conversational knowledge base. It understands, remembers, and retrieves information from multiple documents—allowing you to ask questions naturally and get precise, context-aware answers.

# Features
1:-Conversational Retrieval-Based QA: Uses ConversationalRetrievalChain for intelligent document querying.
2:-Vector Search: Stores and retrieves documents using embeddings.
3:-Streamlit UI: Provides an interactive web-based interface.
4:-Memory Management: Remembers previous interactions for a seamless experience.

# Installation & Setup

1. Clone the Repository:
    https://github.com/aviiiii01/Docu_Mind-chatbot.git
2. Navigate to the Project Directory:
    cd docu_mind-chatbot
3. Create a Virtual Environment (Optional but Recommended):
    python -m venv venv
    source venv/bin/activate   # For macOS/Linux
    venv\Scripts\activate      # For Windows
4. Install Dependencies:
    pip install -r requirements.txt
5. Run the Chatbot:
    streamlit run main.py

# Troubleshooting

1:-ModuleNotFoundError: No module named 'langchain'
    Ensure you installed dependencies with pip install -r requirements.txt.
    Try manually installing LangChain:
        pip install langchain
2:-ModuleNotFoundError: No module named 'langchain_community'
    Ensure you installed dependencies with pip install -r requirements.txt.
    Try manually installing LangChain_community:
        pip install langchain_community

# Contributing

Feel free to fork the repository and submit pull requests for improvements.

# License

This project is licensed under the MIT License.

