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

st.set_page_config(page_title="AgroBot Compact", page_icon="🌾")

# --- 1. CARREGAMENTO DO NOVO MODELO COMPACTO ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('paddydataset.csv')
        df.columns = df.columns.str.strip()
        
        # DEFINIR AS 12 VARIÁVEIS EXATAS DO TREINO
        features = [
            'Hectares', 'Variety', 'Soil Types', 
            'Seedrate(in Kg)', 'Nursery area (Cents)', 'LP_Mainfield(in Tonnes)',
            'DAP_20days', 'Urea_40Days', 'Potassh_50Days', 'Micronutrients_70Days',
            'Weed28D_thiobencarb', 'Pest_60Day(in ml)'
        ]
        
        # Filtra o dataset para recriar o preprocessor igualzinho
        X = df[features]
        
        # Recria preprocessor
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['number']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        preprocessor.fit(X)
        
        # Carrega o modelo NOVO
        # ATENÇÃO: Verifique se o nome do arquivo no GitHub é este mesmo
        model = load_model('modelo_paddy_compacto.h5')
        
        valid_soils = df['Soil Types'].unique().tolist()
        valid_varieties = df['Variety'].unique().tolist()
        
        # Médias APENAS para essas 12 variáveis (caso o usuário não saiba alguma)
        defaults = {}
        for col in X.columns:
            if col in cat_cols: defaults[col] = X[col].mode()[0]
            else: defaults[col] = X[col].mean()
            
        return model, preprocessor, valid_soils, valid_varieties, defaults
        
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")
        return None, None, [], [], {}

model, preprocessor, soils_list, varieties_list, defaults = load_resources()

# --- 2. CONFIGURAÇÃO DA LLM ---
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "SUA_KEY_AQUI"
try: client = Groq(api_key=api_key)
except: client = None

# Memória
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou o AgroBot. Minha rede neural foi otimizada para focar no seu MANEJO.\n\nPara começar, me diga: **Hectares, Solo e Variedade**."})

if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {
        # Básico
        "Hectares": None,
        "Soil Types": None,
        "Variety": None,
        # Manejo (Opcionais - usaremos média se não tiver)
        "Seedrate(in Kg)": None,
        "Nursery area (Cents)": None,
        "LP_Mainfield(in Tonnes)": None,
        "DAP_20days": None,
        "Urea_40Days": None,
        "Potassh_50Days": None,
        "Micronutrients_70Days": None,
        "Weed28D_thiobencarb": None,
        "Pest_60Day(in ml)": None
    }

def get_llm_response(user_input, current_data):
    system_prompt = f"""
    Você é um agrônomo digital. Colete dados para previsão de safra (Modelo Compacto).
    
    ESTADO ATUAL (JSON):
    {json.dumps(current_data)}
    
    LISTAS: Solos={soils_list}, Variedades={varieties_list}

    MISSÃO:
    1. Prioridade Total: Hectares, Solo, Variedade.
    2. Secundário (Manejo): Pergunte sobre Fertilizantes (Ureia, DAP, Potássio), Sementes ou Defensivos.
    3. Se o usuário não souber os detalhes técnicos (Manejo), aceite e diga que usará a média padrão.
    
    RETORNE JSON:
    {{
        "updated_data": {{...}},
        "response_text": "...",
        "ready_to_calculate": true/false (true se tiver pelo menos Hectares, Solo e Variedade e já tiver tentado coletar o resto)
    }}
    """
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

# --- 3. INTERFACE ---
st.title("🌾 AgroBot: Modelo Otimizado")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ex: 5 ha, argiloso, Ponmani. Usei 100kg de Ureia."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if client:
        with st.spinner("Analisando manejo..."):
            ai = get_llm_response(prompt, st.session_state.extracted_data)
            st.session_state.extracted_data = ai["updated_data"]
            
            bot_msg = ai["response_text"]
            
            if ai.get("ready_to_calculate"):
                # Monta o input final misturando Usuário + Médias
                final_input = defaults.copy()
                for k, v in st.session_state.extracted_data.items():
                    if v is not None: final_input[k] = v
                
                # Previsão
                df_in = pd.DataFrame([final_input])
                # Garante numérico
                for col in df_in.columns:
                    if col not in ['Agriblock', 'Variety', 'Soil Types', 'Nursery', 'Wind Direction']: # Lista segura de textos
                        df_in[col] = pd.to_numeric(df_in[col], errors='ignore')

                X_final = preprocessor.transform(df_in)
                if hasattr(X_final, "toarray"): X_final = X_final.toarray()
                
                pred = model.predict(X_final)[0][0]
                
                bot_msg += f"\n\n🎯 **PREVISÃO OTIMIZADA:**\nEstimativa: **{pred:,.2f} Kg**"
                with st.expander("Ver variáveis do Modelo Compacto"):
                    st.write(final_input)

            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
            st.chat_message("assistant").write(bot_msg)
