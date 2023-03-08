#requirements: arxiv, feedparser, requests, sentence-transformers, pinecone

#imports
import feedparser
import requests
from sentence_transformers import SentenceTransformer
import pinecone
import pandas as pd
import arxiv

class ArxivToPinecone:
    def __init__(self, index_name, dimension):
        self.index_name = index_name
        self.dimension = dimension
        
    def get_arxiv_results(self, query):
        search = arxiv.query(
          query = query,
          max_results = 10,
          sort_by = arxiv.SortCriterion.SubmittedDate
        )
        return search
    
    def get_embeddings(self, abstracts):
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        embeddings = model.encode(abstracts, show_progress_bar=True)
        return embeddings
    
    def initialize_pinecone_client(self):
        pinecone.init(api_key="")
        indexes = pinecone.list_indexes()
        if self.index_name not in indexes:
            pinecone.create_index(index_name=self.index_name, dimension=self.dimension)
        pinecone_index = pinecone.Index(index_name=self.index_name)
        return pinecone_index
    
    def upsert_to_pinecone_index(self, records):
        pinecone_index = pinecone.Index(index_name=self.index_name)
        ids, embeddings, metadata = [], [], []
        for record in records:
            ids.append(record[0])
            embeddings.append(record[1])
            metadata.append(record[2])
        pinecone_index.upsert(ids=ids, embeddings=embeddings, metadata=metadata)
    
    def arxiv_to_pinecone(self, query):
        results = self.get_arxiv_results(query)
        abstracts = [r.summary for r in results]
        embeddings = self.get_embeddings(abstracts)
        records = [(r.url, embeddings[i], {'title': r.title, 'authors': r.authors, 'published': r.published, 'categories': r.categories, 'doi': r.doi})
                   for i, r in enumerate(results)]
        pinecone_index = self.initialize_pinecone_client()
        self.upsert_to_pinecone_index(records)
#changed