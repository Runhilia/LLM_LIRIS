import streamlit as st
import time 
import rag2

###### FONCTIONS ######

# Préparation des embeddings
def prepare_embeddings():
    return rag2.prepare_embeddings()

# Génération d'une réponse à partir d'un prompt        
def generate_response(prompt_input, embeddings):
    return rag2.generate_response(prompt_input, embeddings)

###### PAGE ######

st.set_page_config(page_title='LLM - LIRIS', page_icon='💬')
st.title('LLM - LIRIS')

embeddings = prepare_embeddings()

# Initialise l'historique du chat
if 'messages' not in st.session_state.keys():
    st.session_state.messages = [{'role': 'assistant', 'content': 'Hi, how can I help you ?'}]
    
# Affiche les messages depuis l'historique
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Bar pour ecrire un prompt
if prompt := st.chat_input(placeholder='Type a message...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.write(prompt)  

# Reaction au dernier prompt utilisateur
if st.session_state.messages[-1]['role'] != 'assistant':
    with st.chat_message('assistant'):
        with st.spinner('Thinking...'):
            response = generate_response(prompt, embeddings)
            placeholder = st.empty()
            full_response = ''
    for item in response:
        full_response += item
        time.sleep(0.01)
        placeholder.markdown(full_response)
    message = {'role': 'assistant', 'content': full_response}
    st.session_state.messages.append(message)
    
