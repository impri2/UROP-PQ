import json

from rank_bm25 import BM25Okapi

from gpt import get_gpt_response
from dataset import load_beir_dataset
KEYWORD_EXTRACTION_PROMPT = "given the document, extract the queries, in a JSON list format. document:{document}\nkeywords:"
CONTEXT_QUERY_PROMPT = "given the context and the document, generate a query for the document.\ncontext:{context}\ndocument:{document}\nquery:"
corpus = load_beir_dataset('trec-covid','test')['corpus']
corpus_text = [corpus[document]['text'] for document in corpus]
corpus_tokenized = [document.split(" ") for document in corpus_text]
bm25 = BM25Okapi(corpus_tokenized)
def get_keywords(document,trial=3):
    if trial < 1:
        return None
    keywords = get_gpt_response(KEYWORD_EXTRACTION_PROMPT.format(document=document))
    try:
        keywords=json.loads(keywords)
        return keywords
    except:
        keywords = get_keywords(document,trial-1)
        return keywords
    

for i,document in enumerate(corpus_text):
    if i>=10:break
    keywords = get_keywords(document)
    if keywords == None:
        print('keyword extraction error')
        print()
        continue
    retrieved_documents = bm25.get_top_n(keywords,corpus_text,n=3)
    context = '\n\n'.join(retrieved_documents)
    query = get_gpt_response(CONTEXT_QUERY_PROMPT.format(context=context,document=document))
    print("document: ",document)
    print("keyword: ",keywords)
    print('context: ',context)
    print('second query: ',query)
    print()

