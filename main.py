from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from IPython.display import display, HTML
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-N5Z1QEsDO7KlVtCWMj9AT3BlbkFJBgfw3JHS3r7MbTd0SHqG"

loader = TextLoader('C:/Users/dinos/OneDrive/Desktop/documentation.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=20000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

chroma_db_dir = 'chroma_db'
if not os.path.exists(chroma_db_dir):
    os.makedirs(chroma_db_dir)
persist_directory = chroma_db_dir

embeddings = OpenAIEmbeddings()

chroma_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

llm = ChatOpenAI(model_name='gpt-4-1106-preview', max_tokens=4000, temperature=0)
qa_chain = load_qa_chain(llm, chain_type="map_reduce")
retriever = chroma_db.as_retriever(search_kwargs={'k': 2})
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

ef ask_question(question):
    # Display the original question
    display(HTML(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><p><strong>Original Question:</strong> {question}</p></div>"))
    
    sub_questions_response = llm.invoke(f"Edit this question so that it is relevant to FlutterFlow.This is the qestions: {question} - return only edited question")
    sub_questions = sub_questions_response.content.split('\n')[:1]
    
    # Display the sub-questions
    display(HTML(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><p><strong>Sub-questions:</strong></p><ul>{''.join(['<li>' + sub_question + '</li>' for sub_question in sub_questions])}</ul></div>"))
    
    relevant_info = []
    for sub_question in sub_questions:
        sub_question = sub_question.strip()
        if sub_question:
            info = qa.run(sub_question + " and provide as more detail as posible and longest posible anwer")
            relevant_info.append(f"Sub-question: {sub_question} and provide as more detail as posible and longest posible anwer \nRelevant Information: {info}\n")
            # Display the relevant information in real-time
            print(relevant_info[-1])
    
    relevant_info_text = '\n'.join([f"Sub-question: {sub_question}  and provide as more detail as posible and longest posible anwer \nRelevant Information: {info}\n" for sub_question, info in zip(sub_questions, [qa.run(sub_question) for sub_question in sub_questions])])
    answer_response = llm.invoke(f"Based on the relevant information retrieved from the FlutterFlow documentation:\n{relevant_info_text}\n\nAnswer the original question in most detaild as possible with all necesery instructions: {question}")
    
    # Display the final answer in real-time
    print(f"Answer: {answer_response.content}")
    
    return answer_response.content

tools = [
    Tool(
        name="FlutterFlow Documentation QA System",
        func=ask_question,
        description="Useful for answering questions related to FlutterFlow based on the documentation."
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

while True:
    question = input("Ask a question about FlutterFlow (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    agent.run(question)
