import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuração da Página
st.set_page_config(page_title="AgroBot - Previsão de Safra", page_icon="🌾", layout="centered")

# --- FUNÇÃO 1: Carregar Modelos e Dados ---
@st.cache_resource
def load_assets():
    try:
        # Carrega o modelo treinado
        model = load_model('modelo_paddy.h5')
        # Carrega o pre-processador (Scaler + OneHotEncoder)
        preprocessor = joblib.load('preprocessor.pkl')
        # Carrega o dataset apenas para pegar as médias e opções
        df_raw = pd.read_csv('paddydataset.csv')
        return model, preprocessor, df_raw
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None

model, preprocessor, df_raw = load_assets()

# --- FUNÇÃO 2: Gerar Linha Base (Médias/Modas) ---
def get_default_input(df):
    """
    Cria um dicionário com valores padrão para todas as colunas.
    - Numéricos: Usa a Média.
    - Categóricos (Texto): Usa a Moda (o valor mais comum).
    """
    defaults = {}
    # Remove a coluna alvo (Yield) pois ela não entra na previsão
    input_cols = df.drop('Paddy yield(in Kg)', axis=1)
    
    for col in input_cols.columns:
        if input_cols[col].dtype == 'object':
            # Pega o valor mais frequente (Ex: Solo mais comum)
            defaults[col] = input_cols[col].mode()[0]
        else:
            # Pega a média (Ex: Chuva média)
            defaults[col] = input_cols[col].mean()
            
    return pd.DataFrame([defaults])

# --- INTERFACE DO CHATBOT ---

# Cabeçalho e Avatar
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/4205/4205906.png", width=80) # Ícone genérico de fazenda
with col_title:
    st.title("AgroBot Inteligente")
    st.caption("Sistema de previsão de colheita baseado em Redes Neurais.")

st.markdown("---")
st.write("👋 Olá! Eu sou seu assistente agrícola. Para prever sua colheita, preciso que você informe alguns dados principais sobre sua plantação. O restante (clima, vento, etc.) eu vou assumir com base na média histórica da região.")

# Formulário de Entrada (Apenas o essencial)
with st.form("prediction_form"):
    st.subheader("📝 Dados da Plantação")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Hectares
        hectares = st.number_input("Tamanho da Área (Hectares)", min_value=1, value=6, step=1)
        
        # Tipo de Solo (Pega as opções únicas do CSV)
        soil_options = df_raw['Soil Types'].unique().tolist()
        soil_type = st.selectbox("Tipo de Solo", soil_options)
        
        # Variedade do Arroz
        variety_options = df_raw['Variety'].unique().tolist()
        variety = st.selectbox("Variedade do Arroz", variety_options)

    with c2:
        # Sementes
        seedrate = st.number_input("Taxa de Sementes (Kg)", min_value=0, value=int(df_raw['Seedrate(in Kg)'].mean()))
        
        # Fertilizantes (Principais)
        st.markdown("**Fertilizantes (Kg)**")
        dap = st.number_input("DAP (20 dias)", min_value=0, value=int(df_raw['DAP_20days'].mean()))
        urea = st.number_input("Ureia (40 dias)", min_value=0.0, value=df_raw['Urea_40Days'].mean())

    # Botão de Enviar
    submitted = st.form_submit_button("🌱 Calcular Previsão da Safra")

# --- LÓGICA DE PREVISÃO ---
if submitted:
    if model is not None:
        # 1. Carregar a linha base com médias (clima, vento, etc)
        input_data = get_default_input(df_raw)
        
        # 2. Substituir pelos valores que o usuário digitou
        input_data['Hectares'] = hectares
        input_data['Soil Types'] = soil_type
        input_data['Variety'] = variety
        input_data['Seedrate(in Kg)'] = seedrate
        input_data['DAP_20days'] = dap
        input_data['Urea_40Days'] = urea
        
        # Nota: As outras 38 colunas (chuva, temperatura, etc) continuam com os valores médios calculados na função get_default_input
        
        try:
            # 3. Pré-processamento (Converter texto em números e escalar)
            # O array gerado aqui já está no formato que a rede neural gosta
            X_final = preprocessor.transform(input_data)
            
            # ATENÇÃO: Se deu erro de 'toarray' no treino, aqui removemos também.
            # Se X_final for matriz esparsa, converte. Se for denso, mantém.
            if hasattr(X_final, "toarray"):
                X_final = X_final.toarray()

            # 4. Previsão
            prediction = model.predict(X_final)
            predicted_yield = prediction[0][0] # Pega o número de dentro do array

            # 5. Exibir Resultado
            st.success("✅ Processamento Concluído!")
            
            st.markdown(f"""
            ### 🌾 Previsão de Colheita:
            # **{predicted_yield:,.2f} Kg**
            
            <small>Este cálculo considera os insumos informados e a média histórica climática da região.</small>
            """, unsafe_allow_html=True)
            
            # --- ÁREA EXPLICATIVA (Para impressionar a professora) ---
            with st.expander("🔍 Ver detalhes técnicos (Input da Rede Neural)"):
                st.write("Estes são os dados completos enviados para a Rede Neural (Usuário + Médias Históricas):")
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Erro durante a previsão: {e}")
            st.write("Verifique se as colunas do CSV de treino são idênticas ao CSV atual.")
    else:
        st.error("Modelo não carregado. Verifique os arquivos na pasta.")