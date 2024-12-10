
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

#from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db

from langchain_openai import AzureChatOpenAI

from openai import OpenAI

LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')

os.environ["AZURE_OPENAI_API_KEY"] = "3ad9dcb037994be2aa8dd55218481651"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fit-prinz.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-07-01-preview"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o"

# Function to get the prompt templates for generating alternative questions and answering based on context
def get_prompt():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt

# Main function to handle the query process
def query(input):
    if input:
        # Initialize the language model with the specified model name
        # llm = ChatOllama(model=LLM_MODEL)

        LLM_MODEL = "llama3.1:latest"

        llmOllama = ChatOllama(
            model=LLM_MODEL,
            #base_url="https://ollama.fit.fraunhofer.de:11434")
            base_url="http://localhost:11434")

        # Get the vector database instance
        db = get_vector_db()
        # Get the prompt templates
        QUERY_PROMPT, prompt = get_prompt()

        llmAzure = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
                )
        
        llmOllama = ChatOllama(
            model=LLM_MODEL)

        llmFIT = ChatOpenAI(base_url="https://ollama.fit.fraunhofer.de/v1",
            model = "mixtral:latest", # Ein verfügbares Modell aus Ollama wählen
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE4ZWYyY2FiLWZiZmEtNDc2Yi1iYzJjLTk0Nzg2Njk2ZWQwYyJ9.3digMEcl2-OFykdxfVovKOx7-QYYsVMQS8ecVCz2dbo") # Den generierten API-Key verwenden

        llm = llmFIT # llmOllama # llmAzure


        print(prompt)

        # Set up the retriever to generate multiple queries using the language model and the query prompt
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(), 
            # lambda prompt_text: client_chat(prompt_text),
            llm,
            prompt=QUERY_PROMPT
        )

        context = retriever.get_relevant_documents(input)

        print("=========================================================")
        print("Retrieved Context", context)
        print("=========================================================")




        print("=========================================================")
        print("Prompt: ", prompt)
        print("=========================================================")

        print("starting chain")

        # Define the processing chain to retrieve context, generate the answer, and parse the output
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm # llm # llmFIT
           # | client_chat2
            | StrOutputParser()
        )

        response = chain.invoke(input)

        return response

    return None

# Helper function to call the OpenAI API directly
def client_chat(prompt):
    try:

        print("==========      client_chat ====================")
        print(prompt)
        myModel = "teuken0.4-instruct-research:7b"
        myModel = "mixtral:latest" # Ein verfügbares Modell aus Ollama wählen
        #myModel = "mistral:7b" # Ein verfügbares Modell aus Ollama wählen

        response = client.chat.completions.create(
            model=myModel, # Ein verfügbares Modell aus Ollama wählen
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        print("=" * len(f"Model: {myModel}")) 
        print(f"Model: {myModel}")
        print("=" * len(f"Model: {myModel}")) 
        print(response.choices[0].message.content)



        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"
    

import re
# Helper function to call the OpenAI API directly
def client_chat2(prompt):
    try:
        print("==========      client_chat 222222222222222222")
        print(prompt)

        print("==========      client_chat 222222222222222222")

        # Regular expression to match content="..."
        pattern = r'content="([^"]+)"'

        # Find all matches
        matches = re.findall(pattern, str(prompt))

        # Print the results
        for match in matches:
            print("==========      MATCH")

            print(match)

        client = OpenAI(base_url="https://ollama.fit.fraunhofer.de/v1",
                    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjE4ZWYyY2FiLWZiZmEtNDc2Yi1iYzJjLTk0Nzg2Njk2ZWQwYyJ9.3digMEcl2-OFykdxfVovKOx7-QYYsVMQS8ecVCz2dbo") # Den generierten API-Key verwenden

        myModel = "teuken0.4-instruct-research:7b"
        myModel = "mixtral:latest" # Ein verfügbares Modell aus Ollama wählen
        #myModel = "mistral:7b" # Ein verfügbares Modell aus Ollama wählen

        print("Starting LLM")

        response = client.chat.completions.create(
            model=myModel, # Ein verfügbares Modell aus Ollama wählen

            messages=[
                {"role": "user", "content": match}
            ]
        )



        print("=" * len(f"Model: {myModel}")) 
        print(f"Model: {myModel}")
        print("=" * len(f"Model: {myModel}")) 
        print(response.choices[0].message.content)

        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"
    
