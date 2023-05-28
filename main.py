"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate


load_dotenv()

def load_chain(model_repo:str="facebook/opt-iml-1.3b" ):
    """Logic for loading the chain you want to use should go here."""
    hub_llm = HuggingFaceHub(
    repo_id=model_repo,
    model_kwargs={'temperature': 0.7, 'max_length': 100}
    )

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Try to answer the following question, if you don't know the answer please say 
        just that you do not know.
        question:
                {question}
        """
    )

    hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
    return hub_chain

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demoy")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""## Educational Demo 
    In this demo we will try to reproduce a question answering app
    using langchain and huggingface hub.
    \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)""")

with col2:
    st.image(image='img.tiff', width=500, caption='hf demo')

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))
    
with col2:
    model = st.selectbox(
        'Which Model?',
        ('google/flan-t5-base', 'facebook/opt-iml-1.3b'))

# if "generated" not in st.session_state:
#     st.session_state["generated"] = []

# if "past" not in st.session_state:
#     st.session_state["past"] = []

def update_text_with_example():
    print ("in updated")
    st.session_state.email_input = "Which is France's Capital?"

st.button("*See An Example*", type='secondary', help="Click to see an example.", on_click=update_text_with_example)

st.markdown("### Your input:")

chain = load_chain(model_repo=model)

def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(user_input)

    st.write(output)
    print(output)

