import streamlit as st
import time 
import RAG_LLM

###### PAGE ######

st.set_page_config(page_title='RAG - LIRIS', page_icon='💬')
st.title('RAG - LIRIS')

# Vérifiez si les embeddings sont déjà dans le state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None


if st.button("Loading embeddings"):
    with st.spinner("Loading embeddings..."):
        st.session_state.embeddings = RAG_LLM.prepare_embeddings()
    st.success("Embeddings Load !")
    
if st.session_state.embeddings is not None:
    st.write("Embeddings disponibles pour poser des questions.")
if st.session_state.embeddings is  None:
    st.warning("Embeddings not loaded yet. Please load them before asking questions.")

# Initialise l'historique du chat
if 'messages' not in st.session_state.keys():
    st.session_state.messages = [{'role': 'assistant', 'content': 'Hi, how can I help you ?'}]
    
# Affiche les messages depuis l'historique
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Barre pour ecrire un prompt
if prompt := st.chat_input(placeholder='Type a message...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.write(prompt)  

# Réaction au dernier prompt utilisateur
if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = RAG_LLM.generate_response(prompt, st.session_state.embeddings)
            placeholder = st.empty()
            full_response = ''
    for item in response:
        full_response += item
        time.sleep(0.01)
        placeholder.markdown(full_response)
    message = {'role': 'assistant', 'content': full_response}
    st.session_state.messages.append(message)
    
