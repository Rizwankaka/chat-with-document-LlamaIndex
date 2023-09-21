import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI 
import openai
from llama_index import SimpleDirectoryReader   


openai.openai_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Chat with your Documents,powered by LlamaIndex")
st.title("Chat with your Documents")

if "messages" not in st.session_state.keys(): # Initialize the chat messages
    st.session_state.messages = [{"role":"assistant", "content":"ask me questions"}]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Documents...."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, systemprompt="You are expert on Pakistan Studies, your job is to provice the valid and relevant answers. Assuming all the queries related to pakistan studies. Keep your answers based on facts, do not hellucinate.")
        service_content = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_content)
        return index
    
index=load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question",
                                   verbose=True)

if prompt := st.chat_input("Your Question"): # Prompt for user input 
# save to chat history
    st.session_state.messages.append({"role":"user", "content":prompt})
  
for message in st.session_state.messages: # Display the chat history
    with st.chat_message(message["role"]):
        st.write(message["content"])   

# if last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role":"assistant", "content":response.response}
            st.session_state.messages.append(message) # Add response to message history