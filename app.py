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

st.set_page_config(page_title="AgroBot Entrevistador", page_icon="📋")

# --- 1. CONFIGURAÇÃO E CARREGAMENTO ---
@st.cache_resource
def load_resources():
    try:
        df = pd.read_csv('paddydataset.csv')
        df.columns = df.columns.str.strip()
        
        # As 12 Variáveis Obrigatórias
        features = [
            'Hectares', 'Variety', 'Soil Types', 
            'Seedrate(in Kg)', 'Nursery area (Cents)', 'LP_Mainfield(in Tonnes)',
            'DAP_20days', 'Urea_40Days', 'Potassh_50Days', 'Micronutrients_70Days',
            'Weed28D_thiobencarb', 'Pest_60Day(in ml)'
        ]
        
        X = df[features]
        
        # Preprocessor
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['number']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        preprocessor.fit(X)
        
        # Carrega Modelo (com compile=False para evitar erros de versão)
        model = load_model('modelo_paddy_compacto.h5', compile=False)
        
        # Listas de Opções para o Chatbot oferecer
        valid_soils = sorted(df['Soil Types'].unique().tolist())
        valid_varieties = sorted(df['Variety'].unique().tolist())
        
        return model, preprocessor, valid_soils, valid_varieties
        
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, [], []

model, preprocessor, soils_list, varieties_list = load_resources()

# --- 2. NOMES AMIGÁVEIS PARA AS PERGUNTAS ---
# Isso ajuda a IA a fazer perguntas que parecem conversa de humano, não de computador
FIELD_INFO = {
    "Hectares": "Tamanho da área (em Hectares)",
    "Soil Types": f"Tipo de Solo (Opções: {', '.join(soils_list)})",
    "Variety": f"Variedade do Arroz (Opções: {', '.join(varieties_list)})",
    "Seedrate(in Kg)": "Quantidade de Sementes (Kg)",
    "Nursery area (Cents)": "Área do Viveiro (em Cents)",
    "LP_Mainfield(in Tonnes)": "Preparação do Campo Principal (Toneladas)",
    "DAP_20days": "Fertilizante DAP aplicado aos 20 dias (Kg)",
    "Urea_40Days": "Ureia aplicada aos 40 dias (Kg)",
    "Potassh_50Days": "Potássio aplicado aos 50 dias (Kg)",
    "Micronutrients_70Days": "Micronutrientes aplicados aos 70 dias (Kg)",
    "Weed28D_thiobencarb": "Herbicida Thiobencarb (Weed28D)",
    "Pest_60Day(in ml)": "Pesticida aplicado aos 60 dias (ml)"
}

# --- 3. MEMÓRIA DA SESSÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu assistente técnico. Para calcular sua safra, preciso fazer algumas perguntas obrigatórias.\n\nVamos começar: **Qual o tamanho da sua área em Hectares?**"})

# Estado dos dados (Tudo começa como None)
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {key: None for key in FIELD_INFO.keys()}

# --- 4. LÓGICA DA IA (GROQ) ---
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "SUA_KEY_AQUI"
try: client = Groq(api_key=api_key)
except: client = None

def get_next_missing_field(data):
    """Descobre qual é o próximo campo que está vazio (None)"""
    for key, value in data.items():
        if value is None:
            return key, FIELD_INFO[key]
    return None, None

def get_llm_response(user_input, current_data):
    # 1. Descobrir o que falta preencher
    missing_key, missing_desc = get_next_missing_field(current_data)
    
    # 2. Montar o Prompt
    system_prompt = f"""
    Você é um Entrevistador Agrícola Rígido.
    
    ESTADO ATUAL DOS DADOS (JSON):
    {json.dumps(current_data)}
    
    OBJETIVO ATUAL:
    O usuário acabou de responder algo. Tente extrair o dado para o campo que estava faltando.
    
    REGRAS DE EXTRAÇÃO:
    - Se o usuário falou de "Soil Types", o valor DEVE ser um destes: {soils_list}. Se for parecido (ex: "barro"), converta para o mais próximo da lista (ex: "clay").
    - Se o usuário falou de "Variety", o valor DEVE ser um destes: {varieties_list}.
    - Para números, extraia o valor numérico.
    
    PRÓXIMO PASSO (IMPORTANTE):
    1. Atualize o JSON com o dado que o usuário forneceu agora.
    2. Verifique qual é o PRÓXIMO campo que ainda está null.
    3. Na sua resposta de texto (response_text), confirme o que entendeu e faça a pergunta do PRÓXIMO campo obrigatório.
    4. Se o campo atual for 'Soil Types' ou 'Variety', VOCÊ É OBRIGADO a listar as opções válidas na pergunta.
    
    FORMATO DE RESPOSTA (JSON):
    {{
        "updated_data": {{...}},
        "response_text": "Entendido, X registrado. Agora, qual é o [Próxima Pergunta]?",
        "is_complete": true/false (true apenas se TODOS os 12 campos tiverem valor)
    }}
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(completion.choices[0].message.content)

# --- 5. INTERFACE DO CHAT ---
st.title("📋 AgroBot: Coleta Obrigatória")

# Barra lateral para mostrar o progresso (Visualização das opções escolhidas)
with st.sidebar:
    st.header("Status da Coleta")
    count_filled = sum(1 for v in st.session_state.extracted_data.values() if v is not None)
    total_fields = len(FIELD_INFO)
    st.progress(count_filled / total_fields)
    st.write(f"Preenchido: {count_filled}/{total_fields}")
    st.divider()
    st.json(st.session_state.extracted_data)

# Renderiza conversa
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input do Usuário
if prompt := st.chat_input("Digite sua resposta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if client:
        with st.spinner("Validando resposta..."):
            ai = get_llm_response(prompt, st.session_state.extracted_data)
            
            # Atualiza os dados
            st.session_state.extracted_data = ai["updated_data"]
            bot_msg = ai["response_text"]
            
            # Verifica se terminou
            if ai.get("is_complete"):
                # Prepara para cálculo
                df_in = pd.DataFrame([st.session_state.extracted_data])
                
                # Conversão de segurança (Texto -> Número)
                for col in df_in.columns:
                    if col not in ['Variety', 'Soil Types']:
                        df_in[col] = pd.to_numeric(df_in[col], errors='coerce').fillna(0)

                # Roda o Modelo
                try:
                    X_final = preprocessor.transform(df_in)
                    if hasattr(X_final, "toarray"): X_final = X_final.toarray()
                    
                    pred = model.predict(X_final)[0][0]
                    
                    bot_msg += f"\n\n🎉 **COLETA CONCLUÍDA!**\nSua estimativa de colheita é: **{pred:,.2f} Kg**"
                except Exception as e:
                    bot_msg += f"\n(Erro no cálculo: {e})"

            # Resposta do Bot
            st.session_state.messages.append({"role": "assistant", "content": bot_msg})
            st.chat_message("assistant").write(bot_msg)
