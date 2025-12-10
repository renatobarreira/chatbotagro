import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Configuração da Página
st.set_page_config(page_title="AgroBot - Previsão de Safra", page_icon="🌾", layout="centered")

# --- FUNÇÃO 1: Carregar Modelos e Dados ---
@st.cache_resource
def load_assets():
    # Verifica arquivos antes de tentar carregar
    files = ['modelo_paddy.h5', 'preprocessor.pkl', 'paddydataset.csv']
    missing = [f for f in files if not os.path.exists(f)]
    
    if missing:
        st.error(f"❌ ARQUIVOS FALTANDO NO GITHUB: {missing}")
        st.info("Verifique se você fez o upload desses arquivos exatos para o repositório.")
        return None, None, None

    try:
        model = load_model('modelo_paddy.h5')
        preprocessor = joblib.load('preprocessor.pkl')
        df_raw = pd.read_csv('paddydataset.csv') # Verifique se no GitHub está minúsculo mesmo!
        return model, preprocessor, df_raw
    except Exception as e:
        st.error(f"❌ Erro crítico ao carregar: {e}")
        return None, None, None

# Tenta carregar
model, preprocessor, df_raw = load_assets()

# --- INTERFACE (Só desenha se carregou tudo) ---
if df_raw is not None and model is not None:
    
    # Cabeçalho
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/4205/4205906.png", width=80)
    with col_title:
        st.title("AgroBot Inteligente")
        st.caption("Sistema de previsão de colheita baseado em Redes Neurais.")

    st.markdown("---")
    st.write("👋 Olá! Eu sou seu assistente agrícola.")

    # --- FUNÇÃO AUXILIAR ---
    def get_default_input(df):
        defaults = {}
        input_cols = df.drop('Paddy yield(in Kg)', axis=1)
        for col in input_cols.columns:
            if input_cols[col].dtype == 'object':
                defaults[col] = input_cols[col].mode()[0]
            else:
                defaults[col] = input_cols[col].mean()
        return pd.DataFrame([defaults])

    # --- FORMULÁRIO ---
    with st.form("prediction_form"):
        st.subheader("📝 Dados da Plantação")
        
        c1, c2 = st.columns(2)
        
        with c1:
            hectares = st.number_input("Tamanho da Área (Hectares)", min_value=1, value=6, step=1)
            # AQUI OCORRIA O ERRO ANTES - Agora só roda se df_raw existir
            soil_options = df_raw['Soil Types'].unique().tolist()
            soil_type = st.selectbox("Tipo de Solo", soil_options)
            
            variety_options = df_raw['Variety'].unique().tolist()
            variety = st.selectbox("Variedade do Arroz", variety_options)

        with c2:
            seedrate_val = int(df_raw['Seedrate(in Kg)'].mean())
            seedrate = st.number_input("Taxa de Sementes (Kg)", min_value=0, value=seedrate_val)
            
            st.markdown("**Fertilizantes (Kg)**")
            dap_val = int(df_raw['DAP_20days'].mean())
            dap = st.number_input("DAP (20 dias)", min_value=0, value=dap_val)
            
            urea_val = float(df_raw['Urea_40Days'].mean())
            urea = st.number_input("Ureia (40 dias)", min_value=0.0, value=urea_val)

        # Botão de Enviar (INDENTAÇÃO CORRETA: DENTRO DO WITH)
        submitted = st.form_submit_button("🌱 Calcular Previsão da Safra")

    # --- LÓGICA DE PREVISÃO ---
    if submitted:
        # Carregar linha base
        input_data = get_default_input(df_raw)
        
        # Substituir valores do usuário
        input_data['Hectares'] = hectares
        input_data['Soil Types'] = soil_type
        input_data['Variety'] = variety
        input_data['Seedrate(in Kg)'] = seedrate
        input_data['DAP_20days'] = dap
        input_data['Urea_40Days'] = urea
        
        try:
            # Transforma
            X_final = preprocessor.transform(input_data)
            
            if hasattr(X_final, "toarray"):
                X_final = X_final.toarray()

            # Previsão
            prediction = model.predict(X_final)
            predicted_yield = prediction[0][0]

            st.success("✅ Processamento Concluído!")
            st.markdown(f"### 🌾 Previsão de Colheita: **{predicted_yield:,.2f} Kg**")
            
            with st.expander("🔍 Ver detalhes técnicos"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Erro no cálculo: {e}")

else:
    # Se caiu aqui, é porque o load_assets falhou e já imprimiu o erro lá em cima
    st.warning("⚠️ O aplicativo parou porque os dados ou o modelo não foram carregados.")
