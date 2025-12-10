import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Configuração da Página
st.set_page_config(page_title="AgroBot - Previsão de Safra", page_icon="🌾", layout="centered")

# --- FUNÇÃO 1: Carregar Modelo e Recriar Preprocessador ---
@st.cache_resource
def load_assets():
    # 1. Verificação de Segurança
    required_files = ['modelo_paddy.h5', 'paddydataset.csv']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        st.error(f"❌ ARQUIVOS FALTANDO NO GITHUB: {missing}")
        return None, None, None

    try:
        # 2. Carregar Dataset
        df_raw = pd.read_csv('paddydataset.csv')
        
        # ⚠️ CORREÇÃO IMPORTANTE: Remove espaços em branco dos nomes das colunas
        # Ex: "Hectares " vira "Hectares"
        df_raw.columns = df_raw.columns.str.strip()

        # 3. Carregar Rede Neural
        model = load_model('modelo_paddy.h5')

        # 4. Recriar o Preprocessador (Fit)
        X = df_raw.drop('Paddy yield(in Kg)', axis=1)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        preprocessor.fit(X)

        return model, preprocessor, df_raw

    except Exception as e:
        st.error(f"❌ Erro crítico ao processar: {e}")
        return None, None, None

# Carrega tudo
model, preprocessor, df_raw = load_assets()

# --- INTERFACE ---
if df_raw is not None and model is not None:
    
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/4205/4205906.png", width=80)
    with col_title:
        st.title("AgroBot Inteligente")
        st.caption("Sistema de previsão de colheita.")

    st.markdown("---")
    st.info("💡 **Dica:** Os limites dos campos numéricos são baseados no histórico da nossa base de dados para garantir uma previsão mais realista.")

    # Função Auxiliar: Preencher com Médias
    def get_default_input(df):
        defaults = {}
        input_cols = df.drop('Paddy yield(in Kg)', axis=1)
        for col in input_cols.columns:
            if input_cols[col].dtype == 'object':
                defaults[col] = input_cols[col].mode()[0]
            else:
                defaults[col] = input_cols[col].mean()
        return pd.DataFrame([defaults])

    # --- FORMULÁRIO COM LIMITES DINÂMICOS ---
    with st.form("prediction_form"):
        st.subheader("📝 Dados da Plantação")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # --- HECTARES ---
            # Pega o min e max real do banco de dados
            min_h = int(df_raw['Hectares'].min())
            max_h = int(df_raw['Hectares'].max())
            mean_h = int(df_raw['Hectares'].mean())
            
            hectares = st.number_input(
                f"Tamanho da Área (Min: {min_h}, Max: {max_h})",
                min_value=min_h,
                max_value=max_h,
                value=mean_h, # Valor inicial é a média
                step=1
            )
            
            # --- SOLO ---
            soil_options = df_raw['Soil Types'].unique().tolist()
            soil_type = st.selectbox("Tipo de Solo", soil_options)
            
            # --- VARIEDADE ---
            variety_options = df_raw['Variety'].unique().tolist()
            variety = st.selectbox("Variedade do Arroz", variety_options)

        with c2:
            # --- SEMENTES ---
            min_seed = int(df_raw['Seedrate(in Kg)'].min())
            max_seed = int(df_raw['Seedrate(in Kg)'].max())
            mean_seed = int(df_raw['Seedrate(in Kg)'].mean())

            seedrate = st.number_input(
                f"Taxa de Sementes (Kg) [{min_seed}-{max_seed}]", 
                min_value=min_seed, 
                max_value=max_seed, 
                value=mean_seed
            )
            
            st.markdown("**Fertilizantes (Kg)**")
            
            # --- DAP ---
            min_dap = int(df_raw['DAP_20days'].min())
            max_dap = int(df_raw['DAP_20days'].max())
            mean_dap = int(df_raw['DAP_20days'].mean())

            dap = st.number_input(
                f"DAP (20 dias) [{min_dap}-{max_dap}]", 
                min_value=min_dap, 
                max_value=max_dap, 
                value=mean_dap
            )
            
            # --- UREIA (Float) ---
            min_urea = float(df_raw['Urea_40Days'].min())
            max_urea = float(df_raw['Urea_40Days'].max())
            mean_urea = float(df_raw['Urea_40Days'].mean())

            urea = st.number_input(
                f"Ureia (40 dias) [{min_urea:.1f}-{max_urea:.1f}]", 
                min_value=min_urea, 
                max_value=max_urea, 
                value=mean_urea,
                step=0.1
            )

        submitted = st.form_submit_button("🌱 Calcular Previsão da Safra")

    # --- LÓGICA DE PREVISÃO ---
    if submitted:
        input_data = get_default_input(df_raw)
        
        # Substitui pelos valores do usuário
        input_data['Hectares'] = hectares
        input_data['Soil Types'] = soil_type
        input_data['Variety'] = variety
        input_data['Seedrate(in Kg)'] = seedrate
        input_data['DAP_20days'] = dap
        input_data['Urea_40Days'] = urea
        
        try:
            X_final = preprocessor.transform(input_data)
            
            if hasattr(X_final, "toarray"):
                X_final = X_final.toarray()

            prediction = model.predict(X_final)
            predicted_yield = prediction[0][0]

            st.success("✅ Processamento Concluído!")
            st.markdown(f"### 🌾 Previsão de Colheita: **{predicted_yield:,.2f} Kg**")
            
            with st.expander("🔍 Ver input técnico"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Erro na previsão: {e}")

else:
    st.warning("⚠️ Aguardando carregamento...")


