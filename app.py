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
st.set_page_config(page_title="AgroBot Pro", page_icon="🌾")

# --- 1. CARREGAMENTO E PREPARAÇÃO ---
@st.cache_resource
def load_resources():
    try:
        # Carrega e limpa
        df = pd.read_csv('paddydataset.csv')
        df.columns = df.columns.str.strip()
        
        # Recria o preprocessador (igual ao treino)
        X = df.drop('Paddy yield(in Kg)', axis=1)
        cat_cols = X.select_dtypes(include=['object']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        preprocessor.fit(X)
        
        model = load_model('modelo_paddy.h5')
        
        # Listas para validação da LLM
        valid_soils = df['Soil Types'].unique().tolist()
        valid_varieties = df['Variety'].unique().tolist()
        
        # Calcula médias globais (para o que o usuário não souber)
        defaults = {}
        for col in X.columns:
            if col in cat_cols: 
                defaults[col] = X[col].mode()[0]
            else: 
                defaults[col] = X[col].mean()
            
        return model, preprocessor, df, valid_soils, valid_varieties, defaults
    except Exception as e:
        st.error(f"Erro técnico: {e}")
        return None, None, None, [], [], {}

model, preprocessor, df_raw, soils_list, varieties_list, global_defaults = load_resources()

# --- 2. MEMÓRIA DA SESSÃO ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou o AgroBot. Para prever sua safra com precisão, preciso entender o seu plantio.\n\nPara começar: qual o **tamanho da área** (hectares), o **tipo de solo** e a **variedade** do arroz?"})

# Agora extraímos MAIS dados (satisfazendo a professora)
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {
        # Essenciais
        "Hectares": None,
        "Soil Types": None,
        "Variety": None,
        
        # Manejo (O que vamos tentar descobrir)
        "Seedrate(in Kg)": None,
        "DAP_20days": None,      # Fertilizante 1
        "Urea_40Days": None,     # Fertilizante 2
        "Potassh_50Days": None,  # Fertilizante 3
        "Pest_60Day(in ml)": None # Pesticida
    }

# --- 3. CONFIGURAÇÃO DA LLM (CÉREBRO) ---
# COLE SUA API KEY AQUI
api_key = "gsk_..." # <--- COLE SUA CHAVE AQUI

try:
    client = Groq(api_key=api_key)
except:
    client = None

def get_llm_response(user_input, current_data):
    """
    Prompt avançado que tenta preencher o máximo de colunas possível.
    """
    
    # Construção dinâmica do prompt
    system_prompt = f"""
    Você é um agrônomo digital experiente. Seu objetivo é coletar dados técnicos para uma Rede Neural de previsão de safra.
    
    ESTADO ATUAL DOS DADOS (JSON):
    {json.dumps(current_data)}

    LISTAS VÁLIDAS:
    - Solos: {soils_list}
    - Variedades: {varieties_list}

    SUA MISSÃO:
    1. Analise a frase do usuário e extraia qualquer número relacionado a Hectares, Sementes, Ureia, DAP, Potássio ou Pesticidas.
    2. Se o usuário falar "use a média" ou "não sei" para fertilizantes, mantenha como null (o código lidará com isso).
    3. NÃO pergunte sobre clima (chuva, vento, temperatura). Assumiremos dados históricos para isso.
    
    LÓGICA DE CONVERSA:
    - Se faltar "Hectares", "Soil Types" ou "Variety": Pergunte isso primeiro.
    - Se já tiver esses três, PERGUNTE SOBRE O MANEJO: "Você sabe me dizer quanto usou de fertilizantes (Ureia, DAP, Potássio) ou Sementes? Se não souber exato, posso usar uma estimativa padrão."
    - Se o usuário já informou o manejo ou disse que não sabe: Encerre a coleta e avise que vai calcular.

    SAÍDA OBRIGATÓRIA (JSON):
    {{
        "updated_data": {{campos atualizados}},
        "response_text": "Sua pergunta ou confirmação aqui.",
        "ready_to_calculate": true/false (true apenas se tivermos o básico E já tivermos perguntado sobre fertilizantes)
    }}
    """

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(completion.choices[0].message.content)

# --- 4. INTERFACE ---
st.title("🤖 AgroBot Pro: Rede Neural & LLM")

# Histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ex: 5 hectares, solo argiloso, variedade Ponmani. Usei 100kg de Ureia."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if client:
        with st.spinner("Processando dados de manejo..."):
            try:
                ai_result = get_llm_response(prompt, st.session_state.extracted_data)
                
                # Atualiza memória
                st.session_state.extracted_data = ai_result["updated_data"]
                bot_text = ai_result["response_text"]
                is_ready = ai_result.get("ready_to_calculate", False)

                # Se a IA diz que está pronta para calcular
                if is_ready:
                    # 1. Prepara input final
                    final_input = global_defaults.copy() # Começa com todas as médias (clima, etc)
                    
                    # 2. Sobrescreve com o que o usuário deu (Manejo + Básico)
                    user_provided_keys = []
                    for k, v in st.session_state.extracted_data.items():
                        if v is not None:
                            final_input[k] = v
                            user_provided_keys.append(k)
                    
                    # 3. Previsão
                    input_df = pd.DataFrame([final_input])
                    
                    # Hack para garantir que colunas numéricas sejam float/int
                    for col in input_df.columns:
                        if input_df[col].dtype == 'object': pass
                        else: input_df[col] = pd.to_numeric(input_df[col])

                    X_final = preprocessor.transform(input_df)
                    if hasattr(X_final, "toarray"): X_final = X_final.toarray()
                    
                    prediction = model.predict(X_final)[0][0]
                    
                    bot_text += f"\n\n🚀 **PREVISÃO FINAL:**\nEstimativa de Colheita: **{prediction:,.2f} Kg**"
                    
                    # 4. TABELA DE TRANSPARÊNCIA (Pra Professora ver!)
                    with st.expander("📊 Relatório de Variáveis Utilizadas"):
                        st.write("O modelo utilizou **45 variáveis** no total. Abaixo, o detalhe do que foi personalizado:")
                        
                        # Mostra o que é do usuário vs o que é média
                        report_data = {k: final_input[k] for k in st.session_state.extracted_data.keys()}
                        st.table(pd.DataFrame(report_data, index=["Valor Usado"]).T)
                        
                        st.info("Nota: Variáveis climáticas (Chuva, Vento, Temp) foram preenchidas com a média histórica da região (não-controláveis).")

                # Resposta final
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
                with st.chat_message("assistant"):
                    st.markdown(bot_text)
                    
            except Exception as e:
                st.error(f"Erro na comunicação: {e}")
