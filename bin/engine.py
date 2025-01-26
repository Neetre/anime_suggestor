import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Union
import xml.etree.ElementTree as ET

from elasticsearch import Elasticsearch
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset

# import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a searchable document with metadata."""
    id: str
    content: str
    metadata: dict
    vector: Optional[np.ndarray] = None


class SearchEngine:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.index: Dict[str, Set[str]] = {}
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.stemmer = PorterStemmer()
        self.cache: Dict[str, List[Document]] = {}
        self.api_keys: Set[str] = set()
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and stem text while removing stopwords."""
        tokens = word_tokenize(text.lower())
        return [
            self.stemmer.stem(token)
            for token in tokens
            if token not in stopwords.words('english')
        ]
    
    async def index_document(self, doc: Document) -> None:
        """Index a document asynchronously."""
        try:
            doc.vector = self.vectorizer.fit_transform([doc.content]).toarray()[0]

            tokens = set(self._tokenize(doc.content))
            for token in tokens:
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(doc.id)
            
            self.documents[doc.id] = doc
            logger.info(f"Indexed document: {doc.id}")
            
        except Exception as e:
            logger.error(f"Error indexing document {doc.id}: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        max_results: int = 10
    ) -> List[Document]:
        """
        Search for documents matching the query with optional filters.
        Supports boolean operators (AND, OR, NOT) and wildcards.
        """
        try:
            cache_key = f"{query}:{json.dumps(filters)}:{max_results}"
            if cache_key in self.cache:
                logger.info("Returning cached results")
                return self.cache[cache_key]

            query = self._process_query(query)
            query_vector = self.vectorizer.transform([query]).toarray()[0]

            results = []
            for doc in self.documents.values():
                if filters and not self._matches_filters(doc, filters):
                    continue
                    
                score = self._calculate_relevance(query_vector, doc)
                if score > 0:
                    results.append((doc, score))

            results.sort(key=lambda x: x[1], reverse=True)
            top_results = [doc for doc, _ in results[:max_results]]

            self.cache[cache_key] = top_results
            
            return top_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    def _process_query(self, query: str) -> str:
        """Process query string to handle boolean operators and wildcards."""
        query = re.sub(r'\bAND\b', '&', query)
        query = re.sub(r'\bOR\b', '|', query)
        query = re.sub(r'\bNOT\b', '~', query)

        query = query.replace('*', '.*')
        
        return query
    
    def _calculate_relevance(self, query_vector: np.ndarray, doc: Document) -> float:
        """Calculate relevance score using cosine similarity and TF-IDF."""
        similarity = cosine_similarity(
            query_vector.reshape(1, -1),
            doc.vector.reshape(1, -1)
        )[0][0]

        length_factor = 1 / math.log(len(doc.content) + 1)
        
        return similarity * length_factor
    
    def _matches_filters(self, doc: Document, filters: Dict) -> bool:
        """Check if document matches all specified filters."""
        for key, value in filters.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True


def load_documents():
    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.it", streaming=True)
    documents = []
    n = 0
    for record in wikipedia["train"]:
        if n >= 1000:
            break
        doc = Document(
            id=n,
            content=record["title"] + "\n" + record["text"],
            metadata={"source": "Wikipedia"}
        )
        documents.append(doc)
        n += 1
    return documents


async def index_all_documents():
    search_engine = SearchEngine()
    search_engine.api_keys.add("test_key")
    documents = load_documents()
    for doc in documents:
        await search_engine.index_document(doc)
    return search_engine


if __name__ == "__main__":
    asyncio.run(index_all_documents())
    logger.info("Indexing complete")
