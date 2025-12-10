import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from groq import Groq
import os
import json

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="AgroBot AI", page_icon="🤖")

# --- 1. CARREGAMENTO DA REDE NEURAL (Igual ao anterior) ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('paddydataset.csv')
        df.columns = df.columns.str.strip() # Limpeza
        
        # Recriar preprocessador
        X = df.drop('Paddy yield(in Kg)', axis=1)
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        preprocessor.fit(X)
        
        model = load_model('modelo_paddy.h5')
        
        # Pega as listas de opções para ensinar a LLM
        valid_soils = df['Soil Types'].unique().tolist()
        valid_varieties = df['Variety'].unique().tolist()
        
        # Médias para fallback
        defaults = {}
        for col in X.columns:
            if col in cat_cols: defaults[col] = X[col].mode()[0]
            else: defaults[col] = X[col].mean()
            
        return model, preprocessor, df, valid_soils, valid_varieties, defaults
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None, [], [], {}

model, preprocessor, df_raw, soils_list, varieties_list, global_defaults = load_resources()

# --- 2. CONFIGURAÇÃO DA SESSÃO (MEMÓRIA DO CHAT) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Estado inicial: saudação do bot
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou o AgroBot. Vou usar minha Rede Neural para prever sua safra. Para começar, me diga: qual o **tamanho da sua área** (em hectares) e qual o **tipo de solo**?"})

# Variáveis que precisamos extrair (Slot Filling)
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {
        "Hectares": None,
        "Soil Types": None,
        "Variety": None,
        "Seedrate(in Kg)": None,
        "DAP_20days": None,
        "Urea_40Days": None
    }

# --- 3. CONFIGURAÇÃO DA LLM (CÉREBRO CONVERSACIONAL) ---
# ⚠️ COLOQUE SUA API KEY AQUI OU NOS SECRETS DO STREAMLIT
# Se for rodar local, troque pelo string direto: api_key = "gsk_..."
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "SUA_CHAVE_GROQ_AQUI_SE_RODAR_LOCAL"

try:
    client = Groq(api_key=api_key)
except:
    st.warning("⚠️ API Key da Groq não configurada. O chat não funcionará.")
    client = None

def get_llm_response(user_input, current_data):
    """
    Esta função manda o que o usuário disse + o que já sabemos para a LLM.
    A LLM deve retornar um JSON com os dados atualizados e uma resposta em texto.
    """
    
    system_prompt = f"""
    Você é um assistente agrícola especialista. Seu objetivo é coletar dados para alimentar uma Rede Neural.
    
    DADOS QUE PRECISAMOS COLETAR (Se for None, pergunte ao usuário):
    1. Hectares (Número inteiro)
    2. Soil Types (Deve ser um destes: {soils_list})
    3. Variety (Deve ser um destes: {varieties_list})
    4. Seedrate(in Kg) (Número, opcional, se não informado assuma a média)
    5. DAP_20days (Número, opcional, fertilizante)
    6. Urea_40Days (Número, opcional, fertilizante)

    O QUE VOCÊ DEVE FAZER:
    Analise a entrada do usuário e o estado atual. Atualize os campos.
    Se faltar informação crítica (Hectares, Soil Types, Variety), pergunte de forma natural e simpática.
    Se o usuário der uma informação aproximada (ex: "solo de barro"), mapeie para a opção válida mais próxima (ex: "clay").

    ESTADO ATUAL DOS DADOS:
    {json.dumps(current_data)}

    SAÍDA OBRIGATÓRIA (JSON PURO):
    Retorne APENAS um JSON com duas chaves:
    "updated_data": {{objeto com os campos atualizados}},
    "response_text": "Sua resposta simpática para o usuário aqui."
    """

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        response_format={"type": "json_object"} # Força sair JSON
    )
    
    return json.loads(completion.choices[0].message.content)

# --- 4. INTERFACE DO CHAT ---
st.title("🤖 AgroBot: Chat com Rede Neural")

# Mostrar histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar entrada do usuário
if prompt := st.chat_input("Ex: Tenho 6 hectares de solo argiloso..."):
    # 1. Mostrar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Processar com a LLM
    if client:
        with st.spinner("Analisando..."):
            ai_result = get_llm_response(prompt, st.session_state.extracted_data)
            
            # Atualiza os dados extraídos
            st.session_state.extracted_data = ai_result["updated_data"]
            bot_text = ai_result["response_text"]

            # 3. Verificar se temos tudo para rodar a Rede Neural
            # Critério: Precisamos pelo menos de Hectares, Solo e Variedade.
            # O resto podemos usar a média se estiver None.
            data = st.session_state.extracted_data
            if data["Hectares"] and data["Soil Types"] and data["Variety"]:
                
                # Prepara os dados para a Rede Neural
                # Cria um dicionário base com as médias globais
                final_input = global_defaults.copy()
                
                # Sobrescreve com o que a LLM extraiu
                for k, v in data.items():
                    if v is not None:
                        final_input[k] = v
                
                # Transforma em DataFrame de 1 linha
                input_df = pd.DataFrame([final_input])
                
                try:
                    # Roda a Rede Neural (Backend)
                    X_final = preprocessor.transform(input_df)
                    prediction = model.predict(X_final)[0][0]
                    
                    bot_text += f"\n\n🎉 **PREVISÃO PRONTA!**\nCom base no que conversamos e na análise da minha Rede Neural, sua colheita estimada é de: **{prediction:,.2f} Kg**."
                    
                    # Opcional: mostrar os dados usados
                    with st.expander("Ver dados técnicos"):
                        st.write(data)
                        
                except Exception as e:
                    bot_text += f"\n(Tentei calcular, mas houve um erro técnico: {e})"

            # 4. Mostrar resposta do Bot
            st.session_state.messages.append({"role": "assistant", "content": bot_text})
            with st.chat_message("assistant"):
                st.markdown(bot_text)
