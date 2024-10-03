from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

# LLM
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    messages=
    [
    MessagesPlaceholder(variable_name="history"),
    SystemMessagePromptTemplate.from_template("You are a helpful assistant. Your task is answering questions."),
    HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Runnable
runnable = prompt | llm

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

while True:
    question = input("Human: ")
    response = with_message_history.invoke({"question": question}, config={"configurable": {"session_id": "0"}})
    print(f"AI: {response}")
    print(store)