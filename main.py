from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from summarizer import LexRankSummarizer
from embedding import TFIDFEmbedder, RoBERTaEmbedder
from stopwords import ROMANIAN_STOPWORDS

app = FastAPI(title="LexRank Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    compression_rate: float = Field(0.3, ge=0.1, le=1.0)
    embedding_type: Literal["tfidf", "roberta"] = Field("tfidf")

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if req.embedding_type == "tfidf":
        embedder = TFIDFEmbedder()
    elif req.embedding_type == "roberta":
        embedder = RoBERTaEmbedder()
    else:
        raise HTTPException(status_code=400, detail="Invalid embedding_type")

    summarizer = LexRankSummarizer(embedder=embedder, stopwords=ROMANIAN_STOPWORDS)
    result_json = summarizer.summarize_to_json(req.text, compression_rate=req.compression_rate)
    return result_json
