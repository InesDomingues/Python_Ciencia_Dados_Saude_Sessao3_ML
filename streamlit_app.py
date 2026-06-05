import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =============================
# Configuration
# =============================
DATASET_FILE = Path("adultos_portugal_streamlit.csv")
DEFAULT_SYNTHETIC_SAMPLES = 5000
DEFAULT_SEED = 42


# =============================
# Helper functions
# =============================
def classificar_imc(valor: float) -> str:
    """Classifica o IMC de acordo com as categorias habituais da OMS."""
    if valor < 18.5:
        return "Baixo peso"
    if valor < 25:
        return "Peso normal"
    if valor < 30:
        return "Excesso de peso"
    return "Obesidade"


def gerar_dataset_sintetico_adultos_portugal(
    n: int = DEFAULT_SYNTHETIC_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gera um dataset sintético simples de adultos em Portugal.

    Nota: os dados são simulados e servem apenas para fins pedagógicos.
    """
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 86, size=n)

    sexo = rng.choice(
        ["Feminino", "Masculino"],
        size=n,
        p=[0.52, 0.48],
    )

    altura_cm = np.zeros(n)
    for i in range(n):
        if sexo[i] == "Feminino":
            altura_cm[i] = rng.normal(loc=162, scale=3.0)
        else:
            altura_cm[i] = rng.normal(loc=174, scale=3.5)

    altura_cm = np.clip(altura_cm, 140, 205)

    categoria_imc = rng.choice(
        ["Baixo peso", "Peso normal", "Excesso de peso", "Obesidade"],
        size=n,
        p=[0.02, 0.38, 0.38, 0.22],
    )

    imc = np.zeros(n)
    for i in range(n):
        if categoria_imc[i] == "Baixo peso":
            imc[i] = rng.normal(loc=17.8, scale=0.7)
        elif categoria_imc[i] == "Peso normal":
            imc[i] = rng.normal(loc=22.5, scale=1.7)
        elif categoria_imc[i] == "Excesso de peso":
            imc[i] = rng.normal(loc=27.2, scale=1.4)
        else:
            imc[i] = rng.normal(loc=33.0, scale=3.2)

    imc = np.clip(imc, 15, 50)

    altura_m = altura_cm / 100
    peso_kg = imc * (altura_m**2)
    peso_kg = peso_kg + rng.normal(loc=0, scale=1.5, size=n)
    peso_kg = np.clip(peso_kg, 40, 180)

    imc_final = peso_kg / (altura_m**2)
    categoria_imc_final = [classificar_imc(x) for x in imc_final]

    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "idade": idade,
            "sexo": sexo,
            "altura_cm": np.round(altura_cm, 1),
            "peso_kg": np.round(peso_kg, 1),
            "imc": np.round(imc_final, 1),
            "categoria_imc": categoria_imc_final,
            "origem": "Sintética",
        }
    )

    return df


def carregar_ou_criar_dataset(n_synthetic_samples: int, seed: int) -> pd.DataFrame:
    """Carrega o CSV existente ou cria um dataset sintético inicial."""
    if DATASET_FILE.exists():
        df = pd.read_csv(DATASET_FILE)

        # Garante compatibilidade com versões antigas do ficheiro.
        if "origem" not in df.columns:
            df["origem"] = np.where(df.index < n_synthetic_samples, "Sintética", "Real")
        return df

    df = gerar_dataset_sintetico_adultos_portugal(n=n_synthetic_samples, seed=seed)
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")
    return df


def guardar_dataset(df: pd.DataFrame) -> None:
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")

def criar_scatter_altura_peso(df: pd.DataFrame, distinguir_origem: bool = True):
    """Cria scatter plot: altura no eixo x, peso no eixo y, sexo nas cores.

    Se distinguir_origem=True:
    - dados sintéticos aparecem como pontos;
    - dados reais aparecem como bolinhas preenchidas.
    """
    cores_sexo = {
        "Feminino": "red",
        "Masculino": "blue",
    }

    fig, ax = plt.subplots(figsize=(5, 4))

    if distinguir_origem:
        for sexo, cor in cores_sexo.items():

            # Dados sintéticos: pontos
            dados_sinteticos = df[
                (df["sexo"] == sexo) & (df["origem"] == "Sintética")
            ]

            if len(dados_sinteticos) > 0:
                ax.scatter(
                    dados_sinteticos["altura_cm"],
                    dados_sinteticos["peso_kg"],
                    c=cor,
                    marker=".",
                    label=f"{sexo} - Sintética",
                    alpha=0.5,
                    s=15,
                )

            # Dados reais: bolinhas preenchidas
            dados_reais = df[
                (df["sexo"] == sexo) & (df["origem"] == "Real")
            ]

            if len(dados_reais) > 0:
                ax.scatter(
                    dados_reais["altura_cm"],
                    dados_reais["peso_kg"],
                    c=cor,
                    marker="o",
                    label=f"{sexo} - Real",
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.4,
                    s=45,
                )

    else:
        for sexo, cor in cores_sexo.items():
            dados = df[df["sexo"] == sexo]
            ax.scatter(
                dados["altura_cm"],
                dados["peso_kg"],
                c=cor,
                label=sexo,
                alpha=0.6,
                edgecolors="none",
            )

    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Altura vs peso por sexo")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# =============================
# Streamlit app
# =============================
st.set_page_config(
    page_title="Dataset de altura e peso",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dataset de altura e peso")
st.write(
    "Esta aplicação permite recolher dados introduzidos pelos estudantes e juntá-los "
    "a um dataset sintético inicial."
)

nr_synthetic_samples = DEFAULT_SYNTHETIC_SAMPLES
seed = DEFAULT_SEED

df = carregar_ou_criar_dataset(nr_synthetic_samples, seed)

# Garante ids válidos, caso o utilizador altere manualmente o CSV.
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))

col_form, col_info = st.columns([1, 1])

with col_form:
    st.subheader("Adicionar nova observação")

    with st.form("student_form"):
        idade = st.number_input(
            "Idade",
            min_value=18,
            max_value=100,
            value=20,
            step=1,
        )

        sexo = st.selectbox(
            "Sexo",
            ["Feminino", "Masculino"],
        )

        altura_cm = st.number_input(
            "Altura (cm)",
            min_value=120.0,
            max_value=220.0,
            value=170.0,
            step=0.1,
        )

        peso_kg = st.number_input(
            "Peso (kg)",
            min_value=35.0,
            max_value=220.0,
            value=70.0,
            step=0.1,
        )

        submitted = st.form_submit_button("Adicionar ao dataset")

    if submitted:
        altura_m = altura_cm / 100
        imc = peso_kg / (altura_m**2)
        categoria_imc = classificar_imc(imc)

        new_id = int(df["id"].max()) + 1 if len(df) > 0 else 1

        new_row = pd.DataFrame(
            [
                {
                    "id": new_id,
                    "idade": int(idade),
                    "sexo": sexo,
                    "altura_cm": round(float(altura_cm), 1),
                    "peso_kg": round(float(peso_kg), 1),
                    "imc": round(float(imc), 1),
                    "categoria_imc": categoria_imc,
                    "origem": "Real",
                }
            ]
        )

        df = pd.concat([df, new_row], ignore_index=True)
        guardar_dataset(df)

        st.success("Observação adicionada com sucesso.")
        st.write(f"IMC: **{imc:.1f}**")
        st.write(f"Categoria de IMC: **{categoria_imc}**")

with col_info:
    st.subheader("Resumo")
    total = len(df)
    n_sinteticas = int((df["origem"] == "Sintética").sum()) if "origem" in df.columns else 0
    n_reais = int((df["origem"] == "Real").sum()) if "origem" in df.columns else max(0, total - int(nr_synthetic_samples))

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total", total)
    metric_col2.metric("Sintéticas", n_sinteticas)
    metric_col3.metric("Reais", n_reais)

    st.write("Distribuição por sexo")
    st.dataframe(df["sexo"].value_counts().rename_axis("sexo").reset_index(name="n"), use_container_width=True)


st.subheader("Scatter plot: altura vs peso")
fig = criar_scatter_altura_peso(df, distinguir_origem=True)
st.pyplot(fig, use_container_width=False)

st.info(
    "Nota: os dados sintéticos servem apenas para fins pedagógicos e teste de código. "
    "Não devem ser usados para inferência científica sobre a população portuguesa."
)
