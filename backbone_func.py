from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,WebBaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS,Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from google import genai
from google.genai import types
import pymupdf,os
from dotenv import load_dotenv
load_dotenv()


def load_fitzpdf(uploaded):
    # uploadedfiles = pymupdf.open(stream=uploaded,filetype="pdf")
    full_text_fitz = []
    for files in uploaded:
        combine_pdfs = pymupdf.open(stream=files.read(),filetype="pdf")
        for page in combine_pdfs:
            full_text_fitz.append(page.get_text())
    return "\n".join(full_text_fitz)

def load_pdfs(path:str)->list:
    load_pdf = DirectoryLoader(path=path,glob="*.pdf",loader_cls=PyPDFLoader,loader_kwargs={"extract_images":False})
    return load_pdf.load()

def full_extracted_text(loaded_content:list)->list: #chunker(full_extracted_text(load_pdfs(path)))
    full_texts = "\n".join([p.page_content for p in loaded_content])
    return full_texts

def chunker(chunks:list,embeddings)->list:
    embedd = SemanticChunker(embeddings=embeddings,breakpoint_threshold_type="percentile",breakpoint_threshold_amount=90,min_chunk_size=200)
    return embedd.split_text(chunks)

def create_vdb(chunk_list:list,embeddings)->list[float]: #vdb_path:str,
    vectordb = FAISS.from_texts(embedding=embeddings,texts=chunk_list)
    # vectordb.save_local(path=vdb_path,name="rag_pdf_faiss_db")
    return vectordb

apikey = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=apikey)
# chat_history = []

config_=types.GenerateContentConfig(temperature=0.6,system_instruction='''You are an Rag Answering bot, I will provide you real query, 5 chunks lists related to that query from an vector database 
    i want you to analyze those list chunks and give a clear comprehensive output for the user, Let the output be not more than 3 sentences of text,ignore /n and \\n as sometime they may come from pdf.''')

flash_35 = "gemini-3.5-flash-lite"
flash_25L = "gemini-2.5-flash-lite"
# flash_25 = "gemini-2.5-flash"

def reform_question(userquestion:str,chat_history:list)->str:
    message = f'''I want you to reform this new question into an understandable standalone searchable question for a 
    RAG similarity search function with the help of chat history'
    'New Question : {userquestion}
    chat History : {chat_history}'''
    try:
        chat = client.chats.create(model=flash_35)
        reformed_question = chat.send_message(message=[message]) 
        return reformed_question.text
    except Exception as e:
        if "503" in str(e):
            stepney_reform_chat = client.chats.create(model=flash_25L)
            stephneys_reformed_question = stepney_reform_chat.send_message(message=[message])
            return stephneys_reformed_question.text
        else:
            return f'Unexpected Error Occured {e}'

def query_with_llm(user_query:str,vectordb:list[float],semanted:list,chat_history)->str:
    try:
        chat = client.chats.create(model=flash_35,config=config_)
        vectorsearch = vectordb.as_retriever(search_kwargs={"search_type":"similarity","k":5})
        bm25 = BM25Retriever.from_texts(semanted)
        hybrid_searcher = EnsembleRetriever(retrievers=[vectorsearch,bm25],weights=[0.7,0.3],c=65)
        hybrid_result = hybrid_searcher.invoke(input=user_query)
        page_content_lst = []
        for i in hybrid_result:
            page_content_lst.append(i)
        if chat_history:
            query = reform_question(user_query,chat_history)
        else:
            query = user_query
        messages = [f'''Quey :{query} \n Chunks :{page_content_lst}''']
        response = chat.send_message(message=messages).text
        chat_history.append({'user':query,'bot':response})
        page_content_lst.clear()
        return response
    except Exception as e:
        if "503" in str(e):
            print("503 Error occured, cycling the stepney model")
            stepney_chat = client.chats.create(model=flash_25L,config=config_)
            stepney_responce = stepney_chat.send_message(message=messages).text
            chat_history.append({'user': query, 'bot': stepney_responce})
            page_content_lst.clear()
            return stepney_responce
        else:
            return f'Unexpected Error Occured {e}'
#chat st session
def webloader(link:str):
    webload = WebBaseLoader(web_path=link,verify_ssl=True,requests_per_second=3,raise_for_status=1)
    webloaded_contents = []
    for chunks in webload.load():
        webloaded_contents.append(chunks)
    scraped = webloaded_contents[0]
    return scraped

def recursive_conversation(real_vector_db,seman_chunker):
    chat=True
    while chat==True:
        user_question = input('Your Question [type exit to close]: ')
        if 'exit' not in user_question:
            print(query_with_llm(user_question,real_vector_db,seman_chunker))
        else:
            print('Exiting keenly')
            chat=False
            break

def main():
    userinp = input("Enter your Question or a link : ")
    if userinp.startswith("http"):
        extracted_texts_full = webloader(userinp).page_content
        # Content = web_Content[0]
        print('Going with webscraper')
    else:
        loadpdfs = load_pdfs(path="D:\\Winter_Arc\\vscodes\\lang\\RagBot_pdf\\pdf")
        print('Going with PDF extraction')
        extracted_texts_full = full_extracted_text(loaded_content=loadpdfs)
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    seman_chunker = chunker(extracted_texts_full,embeddings)
    real_vector_db=create_vdb(chunk_list=seman_chunker,embeddings=embeddings) 
    # hybrid(real_vector_db,seman_chunker)
    print(recursive_conversation(real_vector_db,seman_chunker))
    # print(query_with_llm(userinp,real_vector_db,seman_chunker))


if __name__ =='__main__':
    main()
