from rank_bm25 import BM25Okapi

from gpt import get_gpt_response
from dataset import load_beir_dataset
QUERY_FIRST_PROMPT = "given the document, write a query. Make the query specific. document:{document}\nquery: "
QUERY_REWRITE_PROMPT = "You will be given a document-query pair. rewrite the query using the context information so the query looks very different and has same meaning. context:{context}\ndocument:{document}\nquery:{query}\n"

corpus = load_beir_dataset('trec-covid','test')['corpus']
corpus_text = [corpus[document]['text'] for document in corpus]
corpus_tokenized = [document.split(" ") for document in corpus_text]
bm25 = BM25Okapi(corpus_tokenized)
for i,document in enumerate(corpus_text):
    if i>=10:break
    query_first = get_gpt_response(QUERY_FIRST_PROMPT.format(document=document))
    retrieved_documents = bm25.get_top_n(query_first.split(" "),corpus_text,n=3)
    context = '\n'.join(retrieved_documents)
    query = get_gpt_response(QUERY_REWRITE_PROMPT.format(context=context,query=query_first,document=document))
    print("document: ",document)
    print('first query: ',query_first)
    print('context: ',context)
    print('second query: ',query)
    print()

