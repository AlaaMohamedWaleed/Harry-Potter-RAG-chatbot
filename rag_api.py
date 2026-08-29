"""Simple Harry Potter RAG API exercise.

Task: complete every TODO in this file, then run the API and test /query.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


# TASK
# Complete TODO 1, TODO 2, TODO 3, and TODO 4.
# Then run the API and test retrieve, chitchat, and off-topic questions.


# ============================= Setup =============================
load_dotenv()

app = FastAPI(title="Harry Potter RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO 1: Add these values to your .env file and load them here.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small", # for embedding the data the small version were used 
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K = int(os.getenv("TOP_K"))

model = SentenceTransformer(EMBEDDING_MODEL) # HERE WE NEED TO LOAD THE SAME EMBEDDING MODEL THAT WE USED TO CREATE THE VECTOR DATABASE, WITH THE SAME DIMENSIONALITY.

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# IF YOU WILL USE ANOTHER LLM FROM ANOTHER PROVIDER, USE THE CORRECT CLASS FROM LANGCHAIN AND PROVIDE THE REQUIRED PARAMETERS.
gemini_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0,
)

groq_llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)


# =========================== Schemas ===========================

class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    book_name: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[Source]


# =========================== Endpoints ===========================

@app.get("/")
def root():
    return {"name": "Harry Potter RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):


    # TODO 2: Write a prompt that returns only one of these words:
    # retrieve, chitchat, or off-topic.

    ROUTER_SYSTEM_PROMPT = """You are a query router for a Harry Potter Q&A system.

    Classify the user's message into exactly one category. Reply with only one word: retrieve, chitchat, or off-topic.

    - retrieve: the user is asking a factual question about the Harry Potter books (characters, plot, spells, places, events, objects, etc.) that requires looking up information from the books.
    - chitchat: the user is greeting you, making small talk, thanking you, or asking something conversational and lighthearted about Harry Potter that doesn't require looking anything up (e.g. "hi", "who's your favorite character?", "do you like Harry Potter?").
    - off-topic: the user is asking about anything unrelated to Harry Potter (other books, general knowledge, coding help, etc.).

    Reply with only one word, lowercase, no punctuation, no explanation."""



    route = groq_llm.invoke([
        SystemMessage(content=(ROUTER_SYSTEM_PROMPT)),
        HumanMessage(content=request.query),
    ]).text.strip().lower()

    if route not in {"retrieve", "chitchat", "off-topic"}:
        route = "off-topic"

    if route == "chitchat":

        # TODO 3: Write a short, friendly prompt for Harry Potter chitchat.
        CHITCHAT_SYSTEM_PROMPT = """You are a friendly Harry Potter fan chatbot. 
        Respond warmly and briefly (1-2 sentences) to greetings, thanks, or casual small talk about the Harry Potter books.
         Do not state specific plot facts, character details, or quotes — keep it light and conversational only,
        be concise."""

        response = groq_llm.invoke([
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=request.query),
        ])

        return QueryResponse(
            query=request.query,
            route=route,
            answer=response.text,
            sources=[],
        )

    if route == "off-topic":
        return QueryResponse(
            query=request.query,
            route=route,
            answer="I can only answer questions about the Harry Potter books.",
            sources=[],
        )

    query_vector = model.encode(
        [f"query: {request.query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    ).points

    context = "\n\n".join(
        f"Book: {result.payload['book_name']}\n"
        f"Page: {result.payload['page_number']}\n"
        f"Content: {result.payload['content']}"
        for result in results
    )

    # TODO 4: Write a prompt that answers only from the provided context.
    # Tell the model to say "I do not know" when the context is not enough.
    RAG_SYSTEM_PROMPT = """You are a Harry Potter expert assistant. Answer the user's question using ONLY the information in the provided context below.
    Rules:
    - Base your answer strictly on the context. Do not use outside knowledge or make anything up.
    - If the context does not contain enough information to answer the question, respond exactly with: "I do not know."
    - Be concise and accurate.
    - You may mention which book/page the information came from if it helps clarify the answer, but do not fabricate citations beyond what is given in the context."""

    response = gemini_llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion:\n{request.query}"
        ),
    ])

    return QueryResponse(
        query=request.query,
        route=route,
        answer=response.text,
        sources=[
            Source(
                book_name=result.payload["book_name"],
                page_number=result.payload["page_number"],
                score=result.score,
            )
            for result in results
        ],
    )
