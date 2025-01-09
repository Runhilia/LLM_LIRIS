import streamlit as st
import time 
import rag2

import json
from googletrans import Translator
import asyncio


###### FONCTIONS ######

# Fonction pour traduire le contenu pertinent
async def translate_json_to_french(input_file, output_file):


    # Charger, traduire et écrire les données en une seule fonction
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialiser le traducteur
    async with Translator() as translator:
        i = 0
        print(len(data))
        for entry in data:
            print(i)
            i += 1
            for key, value in entry.items():
                if key in ['abstract_s', 'keyword_s', 'extracted_keywords']:
                    translateList = []
                    if isinstance(value, list):
                        for elem in value:
                            elem = (await translator.translate(elem, src='fr', dest='en')).text
                            translateList.append(elem)
                    entry[key] = translateList
                else:
                    entry[key] = value

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



# Préparation des embeddings
def prepare_embeddings():
    return rag2.prepare_embeddings()

# Génération d'une réponse à partir d'un prompt        
def generate_response(prompt_input, embeddings):
    return rag2.generate_response(prompt_input, embeddings)

###### PAGE ######

st.set_page_config(page_title='LLM - LIRIS', page_icon='💬')
st.title('LLM - LIRIS')



# asyncio.run(translate_json_to_french('./data/documentsExtractedKeywords.json', './data/translateExtractedKeywords.json'))
rag2.extract_membres("./data/equipes.json", "./data/membres.json")


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
    
