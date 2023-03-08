#requirements: feedparser, requests, sentence-transformers, pinecone, arxiv

# imports
from arvix_search_index import ArxivToPinecone


# code
if __name__ == '__main__':
    index = ArxivToPinecone('arxiv_index', 768)
    index.arxiv_to_pinecone('deep learning')