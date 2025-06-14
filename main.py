from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from summarizer import LexRankSummarizer
from embedding import TFIDFEmbedder, BERTEmbedder
from stopwords import ROMANIAN_STOPWORDS

app = FastAPI(title="LexRank Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10)
    compression_rate: float = Field(0.3, ge=0.1, le=1.0)
    embedding_type: Literal["tfidf", "bert"] = Field("tfidf")

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if req.embedding_type == "tfidf":
        embedder = TFIDFEmbedder()
    elif req.embedding_type == "bert":
        embedder = BERTEmbedder()
    else:
        raise HTTPException(status_code=400, detail="Invalid embedding_type")

    summarizer = LexRankSummarizer(embedder=embedder, stopwords=ROMANIAN_STOPWORDS)
    result_json = summarizer.summarize_to_json(req.text, compression_rate=req.compression_rate)
    print(result_json)
    return result_json
