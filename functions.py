import pdfplumber
from pathlib import Path
import pandas as pd
from operator import itemgetter
import json
import tiktoken
import chromadb
import openai
import ast
import re
import json
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sentence_transformers import CrossEncoder, util


global api_key;
api_key = "api key here";

global chroma_data_path;
chroma_data_path = "database";

# Function to check whether a word is present in a table or not for segregation of regular text and tables
def check_bboxes(word, table_bbox):
    # Check whether word is inside a table bbox.
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

# Function to extract text from a PDF file.
def extract_text_from_pdf(pdf_path):
    p = 0
    full_text = []


    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_no = f"Page {p+1}"
            text = page.extract_text() if page.extract_text() else "No text found"
            tables = page.find_tables()
            table_bboxes = [i.bbox for i in tables]
            tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
            non_table_words = [word for word in page.extract_words() if not any(
                [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
            lines = []

            for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5):

                if 'text' in cluster[0]:
                    try:
                        lines.append(' '.join([i['text'] for i in cluster]))
                    except KeyError:
                        pass

                elif 'table' in cluster[0]:
                    lines.append(json.dumps(cluster[0]['table']))

            if any(lines):
                full_text.append([page_no, " ".join(lines)])
            p +=1

    return full_text

def process_pdf(pdf_path):
    pdf_directory = Path(pdf_path)
    data = []
    for pdf_path in pdf_directory.glob("*.pdf"):
        extracted_text = extract_text_from_pdf(pdf_path)
        extracted_text_df = pd.DataFrame(extracted_text, columns=['Page No.', 'Page_Text'])
        extracted_text_df['Document Name'] = pdf_path.name
        data.append(extracted_text_df)

    finance_pdfs_data = pd.concat(data, ignore_index=True)   
    # Remove rows where 'Page_Text' is null or empty
    finance_pdfs_data = finance_pdfs_data[finance_pdfs_data['Page_Text'].str.strip().astype(bool)]
    finance_pdfs_data['Metadata'] = finance_pdfs_data.apply(lambda x: {'filing_name': x['Document Name'][:-4], 'Page_No.': x['Page No.']}, axis=1)
    # Print a message to indicate all PDFs have been processed
    return finance_pdfs_data;

def create_collection(pdf_path):
    finance_pdfs_data = process_pdf(pdf_path);
    openai.api_key = api_key
    client = chromadb.PersistentClient(path=chroma_data_path)
    model = "text-embedding-ada-002"
    embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=model)
    financedata_collection = client.get_or_create_collection(name='HDFC_Insurance', embedding_function=embedding_function)
    documents_list = finance_pdfs_data["Page_Text"].tolist()
    metadata_list = finance_pdfs_data['Metadata'].tolist()
    financedata_collection.add(
        documents= documents_list,
        ids = [str(i) for i in range(0, len(documents_list))],
        metadatas = metadata_list
    )
    cache_collection = client.get_or_create_collection(name='Finance2_Cache', embedding_function=embedding_function)
    return financedata_collection, cache_collection

def process_query(query, financedata_collection, cache_collection):
    ## Quickly checking the results of the query
    results = financedata_collection.query(
        query_texts=query,
        n_results=10
    )

    cache_results = cache_collection.query(
        query_texts=query,
        n_results=1
    )
    threshold = 0.2
    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()
    # If the distance is greater than the threshold, then return the results from the main collection.

    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
        # Query the collection against the user query and return the top 10 results
        results = financedata_collection.query(
        query_texts=query,
        n_results=10
        )

        # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
        # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
        Keys = []
        Values = []

        for key, val in results.items():
            if key not in ['embeddings', 'uris','data']:
                for i in range(10):
                    Keys.append(str(key)+str(i))
                    Values.append(str(val[0][i]))


        cache_collection.add(
            documents= [query],
            ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
            metadatas = dict(zip(Keys, Values))
        )

        result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
        results_df = pd.DataFrame.from_dict(result_dict)
        results_df


    # If the distance is, however, less than the threshold, you can return the results from cache

    elif cache_results['distances'][0][0] <= threshold:
        cache_result_dict = cache_results['metadatas'][0][0]

        # Loop through each inner list and then through the dictionary
        for key, value in cache_result_dict.items():
            if 'ids' in key:
                ids.append(value)
            elif 'documents' in key:
                documents.append(value)
            elif 'distances' in key:
                distances.append(value)
            elif 'metadatas' in key:
                metadatas.append(value)

        # Create a DataFrame
        results_df = pd.DataFrame({
            'IDs': ids,
            'Documents': documents,
            'Distances': distances,
            'Metadatas': metadatas
        })

    cache_results = cache_collection.query(
        query_texts=query,
        n_results=1
    )

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    results_df['Reranked_scores'] = cross_rerank_scores
    top_3_semantic = results_df.sort_values(by='Distances')
    top_3_rerank = results_df.sort_values(by='Reranked_scores', ascending=False)
    top_3_RAG = top_3_rerank[["Documents", "Metadatas"]][:3]
    print(top_3_RAG)
    if not top_3_RAG.empty:
        retrieved = top_3_RAG.iloc[0]['Documents']  # Access the document with the highest reranked score
    else:
        retrieved = "No relevant documents found."
   

    messages = [
        {"role":"system", "content":"You are an AI assistant to user."},
        {"role":"user", "content":f"""'{query}'. You have to give answer based on '{retrieved}'. If not found inform user that nothing found and prompt for asking next question """},
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages)
    return response.choices[0].message.content



