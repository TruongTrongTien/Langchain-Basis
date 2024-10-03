from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.chains import LLMChain

from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.0-pro", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
    SystemMessagePromptTemplate.from_template("You are an expert in the linguistics field. Your task is answering questions about linguistics."),
    HumanMessagePromptTemplate.from_template("Given a sentence {sentence}, do the following task: {task}."),
    ]
)

chain = prompt | llm
response = chain.invoke({"task": "translate into Korean", "sentence": "Hello, how are you?"})
print(response)

# chain = LLMChain(llm=llm, prompt=prompt)
# response = chain.invoke({"task": "translate into Vietnamese", "sentence": "Hello, how are you?"})
# print(response.get("text"))

