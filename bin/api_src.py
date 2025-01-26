import json
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
import uvicorn

from engine import Document, SearchEngine


app = FastAPI(title="Advanced Search Engine API")
api_key_header = APIKeyHeader(name="X-API-Key")


@app.post("/index")
async def index_document(
    document: Dict,
    api_key: str = Security(api_key_header)
):
    """Index a new document."""
    if api_key not in search_engine.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    doc = Document(
        id=document["id"],
        content=document["content"],
        metadata=document.get("metadata", {})
    )
    await search_engine.index_document(doc)
    return {"status": "success", "document_id": doc.id}


@app.get("/search")
async def search(
    query: str,
    filters: Optional[str] = None,
    max_results: int = 10,
    api_key: str = Security(api_key_header)
):
    """Search for documents."""
    if api_key not in search_engine.api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
        
    filters_dict = json.loads(filters) if filters else None
    results = await search_engine.search(query, filters_dict, max_results)
    
    return {
        "query": query,
        "results": [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }


search_engine = SearchEngine()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)