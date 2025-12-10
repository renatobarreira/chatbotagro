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
    # 1. Verificação de Segurança dos arquivos
    required_files = ['modelo_paddy.h5', 'paddydataset.csv']
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        st.error(f"❌ ARQUIVOS FALTANDO NO GITHUB: {missing}")
        return None, None, None

    try:
        # 2. Carregar o Dataset (Base de Conhecimento)
        # Atenção: O nome deve ser EXATAMENTE como no GitHub (Maiúsculas/Minúsculas importam!)
        # Se no seu GitHub estiver "PaddyDataset.csv", mude abaixo.
        df_raw = pd.read_csv('paddydataset.csv') 

        # 3. Carregar a Rede Neural
        model = load_model('modelo_paddy.h5')

        # 4. RECRIAR O PREPROCESSADOR "AO VIVO" (Solução do Erro)
        # Isso evita o erro de versão do pickle. Recriamos a regra de transformação aqui mesmo.
        
        # Separar colunas igualzinho ao treino
        X = df_raw.drop('Paddy yield(in Kg)', axis=1)
        
        # Identificar tipos
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Criar e Treinar o Preprocessador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # "Ensina" o preprocessor com os dados do CSV
        preprocessor.fit(X)

        return model, preprocessor, df_raw

    except Exception as e:
        st.error(f"❌ Erro crítico ao processar: {e}")
        return None, None, None

# Tenta carregar tudo
model, preprocessor, df_raw = load_assets()

# --- INTERFACE (Só desenha se carregou tudo com sucesso) ---
if df_raw is not None and model is not None:
    
    # Cabeçalho
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/4205/4205906.png", width=80)
    with col_title:
        st.title("AgroBot Inteligente")
        st.caption("Sistema de previsão de colheita baseado em Redes Neurais.")

    st.markdown("---")
    st.write("👋 Olá! Eu sou seu assistente agrícola. Preencha os dados abaixo para simular a colheita.")

    # --- FUNÇÃO AUXILIAR: Preencher com Médias ---
    def get_default_input(df):
        defaults = {}
        # Removemos o alvo para não dar erro
        input_cols = df.drop('Paddy yield(in Kg)', axis=1)
        
        for col in input_cols.columns:
            if input_cols[col].dtype == 'object':
                # Moda para texto
                defaults[col] = input_cols[col].mode()[0]
            else:
                # Média para números
                defaults[col] = input_cols[col].mean()
        return pd.DataFrame([defaults])

    # --- FORMULÁRIO ---
    with st.form("prediction_form"):
        st.subheader("📝 Dados da Plantação")
        
        c1, c2 = st.columns(2)
        
        with c1:
            hectares = st.number_input("Tamanho da Área (Hectares)", min_value=1, value=6, step=1)
            
            # Opções carregadas do CSV
            soil_options = df_raw['Soil Types'].unique().tolist()
            soil_type = st.selectbox("Tipo de Solo", soil_options)
            
            variety_options = df_raw['Variety'].unique().tolist()
            variety = st.selectbox("Variedade do Arroz", variety_options)

        with c2:
            # Valores padrão (médias) sugeridos no input
            seed_default = int(df_raw['Seedrate(in Kg)'].mean())
            seedrate = st.number_input("Taxa de Sementes (Kg)", min_value=0, value=seed_default)
            
            st.markdown("**Fertilizantes (Kg)**")
            dap_default = int(df_raw['DAP_20days'].mean())
            dap = st.number_input("DAP (20 dias)", min_value=0, value=dap_default)
            
            urea_default = float(df_raw['Urea_40Days'].mean())
            urea = st.number_input("Ureia (40 dias)", min_value=0.0, value=urea_default)

        submitted = st.form_submit_button("🌱 Calcular Previsão da Safra")

    # --- LÓGICA DE PREVISÃO ---
    if submitted:
        # 1. Carregar linha base (médias de clima, vento, etc.)
        input_data = get_default_input(df_raw)
        
        # 2. Substituir pelos valores que o usuário digitou
        input_data['Hectares'] = hectares
        input_data['Soil Types'] = soil_type
        input_data['Variety'] = variety
        input_data['Seedrate(in Kg)'] = seedrate
        input_data['DAP_20days'] = dap
        input_data['Urea_40Days'] = urea
        
        try:
            # 3. Transformar os dados (Usando o preprocessor recém-criado)
            X_final = preprocessor.transform(input_data)
            
            # Garantir formato denso se necessário
            if hasattr(X_final, "toarray"):
                X_final = X_final.toarray()

            # 4. Previsão
            prediction = model.predict(X_final)
            predicted_yield = prediction[0][0]

            # 5. Exibir
            st.success("✅ Processamento Concluído!")
            st.markdown(f"### 🌾 Previsão de Colheita: **{predicted_yield:,.2f} Kg**")
            
            with st.expander("🔍 Ver detalhes técnicos (Input da Rede Neural)"):
                st.write("Dados combinados (Input Usuário + Médias Históricas):")
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Erro durante a previsão: {e}")

else:
    # Se caiu aqui, o load_assets falhou e já mostrou o erro lá em cima.
    st.warning("⚠️ Aguardando carregamento dos dados...")
