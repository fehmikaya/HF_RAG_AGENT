
###   RAG Agent with Langchain and Langgraph, Hallucination and Sanity Checks with Websearch 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import END, StateGraph

from customllama3 import CustomLlama3

from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
import os


class RAGAgent():

    HF_TOKEN = os.getenv("HF_TOKEN")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    if HF_TOKEN is None:
        st.error("API key not found. Please set the HF_TOKEN secret in your Hugging Face Space.")
        st.stop()
    if TAVILY_API_KEY is None:
        st.error("API key not found. Please set the TAVILY_API_KEY secret in your Hugging Face Space.")
        st.stop()

    retrieval_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    answer_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    hallucination_prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    answer_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    def reset_chains():
        RAGAgent.retrieval_grader = RAGAgent.retrieval_grader_prompt | CustomLlama3(bearer_token = RAGAgent.HF_TOKEN) | JsonOutputParser()
        RAGAgent.rag_chain = RAGAgent.answer_prompt | CustomLlama3(bearer_token = RAGAgent.HF_TOKEN) | StrOutputParser()
        RAGAgent.hallucination_grader = RAGAgent.hallucination_prompt | CustomLlama3(bearer_token = RAGAgent.HF_TOKEN) | JsonOutputParser()
        RAGAgent.answer_grader = RAGAgent.answer_grader_prompt | CustomLlama3(bearer_token = RAGAgent.HF_TOKEN) | JsonOutputParser()
       
    def __init__(self, docs):
        print("init")
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embedding_function,
        )
        RAGAgent.retriever = vectorstore.as_retriever()
        RAGAgent.reset_chains()
        RAGAgent.logs=""

    def add_log(log):
        RAGAgent.logs += log + "\n"
        
    web_search_tool = TavilySearchResults(k=3)

    class GraphState(TypedDict):
      question: str
      generation: str
      web_search: str
      documents: List[str]
    
    def retrieve(state):
        RAGAgent.add_log("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = RAGAgent.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        RAGAgent.add_log("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = RAGAgent.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}


    def grade_documents(state):

        RAGAgent.add_log("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "Yes"
        
        for d in documents:
            score = RAGAgent.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                RAGAgent.add_log("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
                web_search = "No"
            # Document not relevant
            else:
                RAGAgent.add_log("---GRADE: DOCUMENT NOT RELEVANT---")
                
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def web_search(state):

        RAGAgent.add_log("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = RAGAgent.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}


    def decide_to_generate(state):

        RAGAgent.add_log("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        web_search = state["web_search"]
        filtered_documents = state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            RAGAgent.add_log("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            RAGAgent.add_log("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):

        RAGAgent.add_log("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = RAGAgent.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            RAGAgent.add_log("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = RAGAgent.answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                RAGAgent.add_log("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                RAGAgent.add_log("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            RAGAgent.add_log("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae

    # Build graph
    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    # Compile
    app = workflow.compile()  