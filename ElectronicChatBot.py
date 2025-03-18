import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from arxiv import Client, Search, SortCriterion
import wikipedia
import faiss
import numpy as np
from typing import Literal, Optional, TypedDict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define CustomerState
class CustomerState(TypedDict):
    query: str
    category: Optional[Literal["Technical", "Billing", "General"]]
    sentiment: Optional[Literal["Positive", "Negative", "Neutral"]]
    response: Optional[str]
    query_type: Optional[Literal["General", "Technical"]]
    search_results: Optional[list]
    embeddings: Optional[np.ndarray]
    escalated: Optional[bool]

# Initialize models
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.IndexFlatL2(384)

# Function to check if query is about electronic devices
def is_electronic_device_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Determine if this query is about electronic devices (e.g., phones, laptops, TVs, etc.). "
            "Respond with only 'Yes' or 'No': {query}"
        )
    )
    response = llm.invoke(prompt.format(query=query))
    return response.content.strip() == "Yes"

def categorize_query(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Classify this customer query into Technical, Billing, or General category: {query}"
    )
    response = llm.invoke(prompt.format(query=query))
    return response.content.strip()

def analyze_sentiment(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Analyze the sentiment of this query as Positive, Negative, or Neutral: {query}"
    )
    response = llm.invoke(prompt.format(query=query))
    return response.content.strip()

def classify_query_type(query):
    query_lower = query.lower()
    if "why" in query_lower or "what" in query_lower or "how" in query_lower:
        return "General"
    else:
        return "Technical"

def wiki_search(query):
    wikipedia.set_lang("en")
    try:
        summary = wikipedia.summary(query, sentences=3)
        return [summary]
    except Exception as e:
        return [f"Wikipedia search failed: {str(e)}"]

def arxiv_search_and_embed(query):
    client = Client()
    search = Search(
        query=query,
        max_results=3,
        sort_by=SortCriterion.Relevance
    )
    results = list(client.results(search))
    abstracts = [result.summary for result in results]
    embeddings = embedding_model.encode(abstracts)
    faiss_index.add(embeddings)
    return abstracts

def generate_response(query, category, sentiment, search_results):
    prompt = PromptTemplate(
        input_variables=["query", "category", "sentiment", "search_results"],
        template=(
            "Generate an informative response.\n"
            "Customer Query: {query}\n"
            "Category: {category}\n"
            "Sentiment: {sentiment}\n"
            "Relevant Information: {search_results}\n"
            "If sentiment is Negative, include: 'This issue has been escalated to our support team for immediate attention.'\n"
            "Response:"
        )
    )
    formatted_results = '\n\n'.join(search_results)
    response = llm.invoke(prompt.format(
        query=query,
        category=category,
        sentiment=sentiment,
        search_results=formatted_results
    ))
    return response.content.strip()

def chatbot_interface(user_query):
    # Check if query is about electronic devices using LLM
    if not is_electronic_device_query(user_query):
        response = "I'm sorry, I can only assist with questions about electronic devices. Please ask about a device like a phone, laptop, or other electronics."
        debug_details = (
            f"💡 **Debug Information**:\n"
            f"- **Category:** N/A\n"
            f"- **Sentiment:** N/A\n"
            f"- **Query Type:** N/A\n"
            f"- **Escalated:** False\n"
            f"- **Search Results:** None (query not related to electronic devices)\n"
        )
        return response, debug_details

    # Process normally if it's an electronic device query
    category = categorize_query(user_query)
    sentiment = analyze_sentiment(user_query)
    query_type = classify_query_type(user_query)

    # Ensure escalated is set correctly based on sentiment
    escalated = (sentiment.lower() == "negative")  # Case-insensitive comparison

    if query_type == "General":
        search_results = wiki_search(user_query)
    else:
        search_results = arxiv_search_and_embed(user_query)

    response = generate_response(user_query, category, sentiment, search_results)

    # Debug information with corrected escalated flag
    debug_details = (
        f"💡 **Debug Information**:\n"
        f"- **Category:** {category}\n"
        f"- **Sentiment:** {sentiment}\n"
        f"- **Query Type:** {query_type}\n"
        f"- **Escalated:** {escalated}\n"
        f"- **Search Results:** {' '.join(search_results)}\n"
    )

    return response, debug_details

# Construct Gradio UI
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, label="Enter your query about an electronic device"),
    outputs=[
        gr.Textbox(label="Chatbot Response"),
        gr.Markdown(label="Internal Debug Info")
    ],
    title="Electronic Devices Chatbot UI",
    description="This chatbot only answers questions about electronic devices (phones, laptops, etc.), with escalation for negative sentiment queries."
)

iface.launch(debug=True)