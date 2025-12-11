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

st.set_page_config(page_title="AgroBot Entrevistador", page_icon="👨‍🌾")

# --- 1. CONFIGURAÇÃO E CARREGAMENTO ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('paddydataset.csv')
        df.columns = df.columns.str.strip()
        
        # Variáveis Técnicas (Internas)
        features = [
            'Hectares', 'Variety', 'Soil Types', 
            'Seedrate(in Kg)', 'Nursery area (Cents)', 'LP_Mainfield(in Tonnes)',
            'DAP_20days', 'Urea_40Days', 'Potassh_50Days', 'Micronutrients_70Days',
            'Weed28D_thiobencarb', 'Pest_60Day(in ml)'
        ]
        
        X = df[features]
        
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['number']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        preprocessor.fit(X)
        
        # Carrega modelo ignorando compilação
        model = load_model('modelo_paddy_compacto.h5', compile=False)
        
        valid_soils = sorted(df['Soil Types'].unique().tolist())
        valid_varieties = sorted(df['Variety'].unique().tolist())
        
        return model, preprocessor, valid_soils, valid_varieties
        
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, [], []

model, preprocessor, soils_list, varieties_list = load_resources()

# --- 2. DICIONÁRIO DE TRADUÇÃO (Beleza na Interface) ---
# Chave = Nome Técnico (CSV) | Valor = Nome Bonito (Tela)
FRIENDLY_NAMES = {
    "Hectares": "Tamanho da Área (ha)",
    "Soil Types": "Tipo de Solo",
    "Variety": "Variedade do Arroz",
    "Seedrate(in Kg)": "Sementes (Kg)",
    "Nursery area (Cents)": "Área do Viveiro (Cents)",
    "LP_Mainfield(in Tonnes)": "Preparo do Solo (Ton)",
    "DAP_20days": "Adubo DAP (Kg)",
    "Urea_40Days": "Ureia (Kg)",
    "Potassh_50Days": "Potássio (Kg)",
    "Micronutrients_70Days": "Micronutrientes (Kg)",
    "Weed28D_thiobencarb": "Herbicida Thiobencarb",
    "Pest_60Day(in ml)": "Pesticida (ml)"
}

# --- 3. CONFIGURAÇÃO DA SESSÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou o AgroBot. Vou te ajudar a prever sua colheita.\n\nPara começar, qual o **tamanho da sua área (em hectares)**?"})

if "extracted_data" not in st.session_state:
    # Cria o dicionário vazio usando as chaves técnicas
    st.session_state.extracted_data = {key: None for key in FRIENDLY_NAMES.keys()}

# --- 4. LÓGICA DA IA ---
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "SUA_KEY_AQUI"
try: client = Groq(api_key=api_key)
except: client = None

def get_next_missing_field(data):
    """Encontra o próximo campo vazio e retorna o nome técnico e o bonito"""
    for key, value in data.items():
        if value is None:
            return key, FRIENDLY_NAMES[key]
    return None, None

def get_llm_response(user_input, current_data):
    # Descobre o que falta perguntar
    missing_key, missing_name = get_next_missing_field(current_data)
    
    system_prompt = f"""
    Você é um Assistente Técnico Agrícola.
    
    ESTADO ATUAL (JSON):
    {json.dumps(current_data)}
    
    OBJETIVO:
    O campo que falta preencher é: '{missing_key}' ({missing_name}).
    Tente extrair essa informação da resposta do usuário.
    
    REGRAS DE EXTRAÇÃO:
    - Se for 'Soil Types', deve ser um destes: {soils_list}.
    - Se for 'Variety', deve ser um destes: {varieties_list}.
    - Se for número, extraia apenas o número.
    
    FLUXO DE CONVERSA:
    1. Atualize o JSON com o dado novo (se o usuário forneceu).
    2. Identifique qual é o PRÓXIMO campo vazio DEPOIS desse.
    3. Na sua resposta de texto, confirme o que anotou e pergunte sobre o PRÓXIMO campo usando o nome amigável ("{missing_name}").
    4. Se for perguntar de Solo ou Variedade, LISTE AS OPÇÕES.
    
    SAÍDA (JSON):
    {{
        "updated_data": {{...}},
        "response_text": "Texto simpático...",
        "is_complete": true/false
    }}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(completion.choices[0].message.content)

# --- 5. INTERFACE (FRONTEND) ---
st.title("👨‍🌾 AgroBot: Previsão de Safra")

# --- BARRA LATERAL MELHORADA (VISUAL AMIGÁVEL) ---
with st.sidebar:
    st.header("📋 Ficha Técnica")
    
    # Calcula progresso
    filled = sum(1 for v in st.session_state.extracted_data.values() if v is not None)
    total = len(FRIENDLY_NAMES)
    st.progress(filled / total)
    st.caption(f"Progresso: {filled}/{total}")
    st.divider()
    
    # Loop para mostrar bonitinho
    for technical_name, friendly_name in FRIENDLY_NAMES.items():
        value = st.session_state.extracted_data.get(technical_name)
        
        if value is not None:
            # Se já preencheu: Verde com Check
            st.markdown(f"✅ **{friendly_name}**")
            st.code(f"{value}") # Mostra o valor em destaque
        else:
            # Se falta: Cinza com Círculo vazio
            st.markdown(f"⚪ {friendly_name}")
            
# --- CHAT ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Digite aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if client:
        with st.spinner("Analisando..."):
            try:
                ai = get_llm_response(prompt, st.session_state.extracted_data)
                st.session_state.extracted_data = ai["updated_data"]
                bot_msg = ai["response_text"]
                
                if ai.get("is_complete"):
                    # Prepara DataFrame para cálculo
                    df_in = pd.DataFrame([st.session_state.extracted_data])
                    
                    # Converte números
                    for col in df_in.columns:
                        if col not in ['Variety', 'Soil Types']:
                            df_in[col] = pd.to_numeric(df_in[col], errors='coerce').fillna(0)

                    # Previsão
                    X_final = preprocessor.transform(df_in)
                    if hasattr(X_final, "toarray"): X_final = X_final.toarray()
                    
                    pred = model.predict(X_final)[0][0]
                    
                    bot_msg += f"\n\n🚜 **RESULTADO DA ANÁLISE:**\nSua estimativa de produção é de **{pred:,.2f} Kg**."
                    st.balloons() # Efeito visual de sucesso!

                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                st.chat_message("assistant").write(bot_msg)
                
                # Força atualização da barra lateral
                st.rerun()
                
            except Exception as e:
                st.error(f"Erro: {e}")
