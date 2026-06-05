from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

        # Compatibilidade com versões antigas do ficheiro.
        if "origem" not in df.columns:
            df["origem"] = np.where(
                df.index < n_synthetic_samples,
                "Sintética",
                "Aluno",
            )

        return df

    df = gerar_dataset_sintetico_adultos_portugal(
        n=n_synthetic_samples,
        seed=seed,
    )
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")
    return df


def guardar_dataset(df: pd.DataFrame) -> None:
    df.to_csv(DATASET_FILE, index=False, encoding="utf-8-sig")


def criar_scatter_altura_peso(df: pd.DataFrame, distinguir_origem: bool = True):
    """Cria scatter plot: altura no eixo x, peso no eixo y, sexo nas cores.

    Se distinguir_origem=True:
    - dados sintéticos aparecem como pontos pequenos;
    - dados dos alunos aparecem como bolinhas preenchidas.

    Inclui uma linha de regressão linear usando todos os dados.
    """
    cores_sexo = {
        "Feminino": "red",
        "Masculino": "blue",
    }

    fig, ax = plt.subplots(figsize=(4, 3))

    if distinguir_origem:
        for sexo, cor in cores_sexo.items():
            # Dados sintéticos: pontos pequenos
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
                    alpha=0.4,
                    s=10,
                )

            # Dados dos alunos: bolinhas preenchidas
            dados_alunos = df[
                (df["sexo"] == sexo) & (df["origem"] == "Aluno")
            ]

            if len(dados_alunos) > 0:
                ax.scatter(
                    dados_alunos["altura_cm"],
                    dados_alunos["peso_kg"],
                    c=cor,
                    marker="o",
                    label=f"{sexo} - Aluno",
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
                s=15,
            )

    # Linha de regressão usando todos os dados
    df_reg = df[["altura_cm", "peso_kg"]].dropna()

    if len(df_reg) >= 2:
        x = df_reg["altura_cm"].to_numpy()
        y = df_reg["peso_kg"].to_numpy()

        coef = np.polyfit(x, y, deg=1)
        linha = np.poly1d(coef)

        x_linha = np.linspace(x.min(), x.max(), 100)
        y_linha = linha(x_linha)

        ax.plot(
            x_linha,
            y_linha,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Regressão linear",
        )

    ax.set_xlabel("Altura (cm)")
    ax.set_ylabel("Peso (kg)")
    ax.set_title("Altura vs peso por sexo")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    return fig


def avaliar_codigo(codigo: str, requisitos: dict) -> tuple[bool, list[str], list[str]]:
    """Avalia se o código do aluno contém elementos esperados."""
    codigo_lower = codigo.lower()

    mensagens_sucesso = []
    mensagens_dicas = []

    for elemento, dica in requisitos.items():
        if elemento.lower() in codigo_lower:
            mensagens_sucesso.append(f"✓ Encontrado: `{elemento}`")
        else:
            mensagens_dicas.append(dica)

    sucesso = len(mensagens_dicas) == 0
    return sucesso, mensagens_sucesso, mensagens_dicas


def mostrar_feedback_codigo(codigo: str, requisitos: dict) -> bool:
    """Mostra feedback automático ao código escrito pelo aluno."""
    if not codigo.strip():
        st.warning("Escreve algum código antes de verificar.")
        return False

    sucesso, mensagens_sucesso, mensagens_dicas = avaliar_codigo(codigo, requisitos)

    for msg in mensagens_sucesso:
        st.success(msg)

    if mensagens_dicas:
        st.warning("Ainda há alguns pontos a melhorar:")
        for dica in mensagens_dicas:
            st.write(f"- {dica}")
        st.info("Revê as dicas e tenta melhorar o código.")
    else:
        st.success("Muito bem. O código contém os elementos principais deste passo.")

    return sucesso


def preparar_dados_modelo(df: pd.DataFrame):
    """Prepara X e y para o exemplo supervisionado."""
    X = df[["idade", "altura_cm", "peso_kg", "imc"]]
    y = df["sexo"]

    return X, y


def executar_pipeline_supervisionado(
    df: pd.DataFrame,
    modelo_nome: str = "Árvore de decisão",
):
    """Executa uma pipeline supervisionada controlada."""
    X, y = preparar_dados_modelo(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    if modelo_nome == "Árvore de decisão":
        modelo = DecisionTreeClassifier(random_state=42)
    elif modelo_nome == "Random forest":
        modelo = RandomForestClassifier(random_state=42)
    else:
        modelo = LogisticRegression(max_iter=1000)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Feminino", "Masculino"],
    )
    report = classification_report(y_test, y_pred)

    return acc, cm, report


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
    "a um dataset sintético inicial. Inclui ainda um modo desafio para praticar "
    "os passos de um exemplo supervisionado em Python."
)

nr_synthetic_samples = DEFAULT_SYNTHETIC_SAMPLES
seed = DEFAULT_SEED

df = carregar_ou_criar_dataset(nr_synthetic_samples, seed)

# Garante ids válidos, caso o utilizador altere manualmente o CSV.
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))

# =============================
# Data collection
# =============================
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
                    "origem": "Aluno",
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
    n_sinteticas = int((df["origem"] == "Sintética").sum())
    n_alunos = int((df["origem"] == "Aluno").sum())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total", total)
    metric_col2.metric("Sintéticas", n_sinteticas)
    metric_col3.metric("Alunos", n_alunos)

    st.write("Distribuição por sexo")
    dist_sexo = df["sexo"].value_counts().rename_axis("sexo").reset_index(name="n")
    st.dataframe(dist_sexo, use_container_width=True)

# =============================
# Scatter plot
# =============================
st.subheader("Scatter plot: altura vs peso")

fig = criar_scatter_altura_peso(df, distinguir_origem=True)
st.pyplot(fig, use_container_width=False)

# =============================
# Dataset
# =============================
st.subheader("Dataset atual")

st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Descarregar dataset em CSV",
    data=csv,
    file_name="adultos_portugal_sintetico_atualizado.csv",
    mime="text/csv",
)

# =============================
# Challenge mode
# =============================
st.divider()

st.header("Modo desafio: exemplo prático supervisionado")

st.write(
    "Neste modo, vais escrever pequenos blocos de código para percorrer os passos "
    "de um exemplo supervisionado: importar dados, definir X e y, dividir treino/teste, "
    "treinar o modelo, avaliar métricas e interpretar."
)

st.info(
    "Por segurança, a app não executa diretamente código livre. Primeiro verifica se o código "
    "contém os elementos essenciais. Quando os passos principais estão corretos, a app executa "
    "uma versão controlada da pipeline."
)

tabs = st.tabs(
    [
        "1. Importar dados",
        "2. Definir X e y",
        "3. Dividir treino/teste",
        "4. Treinar modelo",
        "5. Avaliar métricas",
        "6. Interpretar",
    ]
)

# =============================
# Step 1
# =============================
with tabs[0]:
    st.subheader("1. Importar dados")

    st.write("Objetivo: carregar o dataset para um DataFrame chamado `df`.")

    codigo_1 = st.text_area(
        "Escreve o código para importar os dados:",
        value='df = pd.read_csv("adultos_portugal_streamlit.csv")\ndf.head()',
        height=140,
        key="codigo_1",
    )

    requisitos_1 = {
        "pd.read_csv": "Usa `pd.read_csv(...)` para carregar o ficheiro CSV.",
        "df": "Guarda os dados numa variável chamada `df`.",
        "head": "Usa `df.head()` para visualizar as primeiras linhas.",
    }

    if st.button("Verificar passo 1"):
        mostrar_feedback_codigo(codigo_1, requisitos_1)

    with st.expander("Ver solução possível"):
        st.code(
            'df = pd.read_csv("adultos_portugal_streamlit.csv")\n'
            "df.head()",
            language="python",
        )

# =============================
# Step 2
# =============================
with tabs[1]:
    st.subheader("2. Definir X e y")

    st.write(
        "Objetivo: definir as variáveis preditoras `X` e a variável-alvo `y`. "
        "Neste exemplo, vamos tentar prever `sexo` a partir de idade, altura, peso e IMC."
    )

    codigo_2 = st.text_area(
        "Escreve o código para definir X e y:",
        value='X = df[["idade", "altura_cm", "peso_kg", "imc"]]\ny = df["sexo"]',
        height=140,
        key="codigo_2",
    )

    requisitos_2 = {
        "X": "Define uma variável chamada `X` com as variáveis preditoras.",
        "y": "Define uma variável chamada `y` com a variável-alvo.",
        "sexo": "Neste exemplo, a variável-alvo deve ser `sexo`.",
        "idade": "Inclui `idade` nas variáveis preditoras.",
        "altura_cm": "Inclui `altura_cm` nas variáveis preditoras.",
        "peso_kg": "Inclui `peso_kg` nas variáveis preditoras.",
        "imc": "Inclui `imc` nas variáveis preditoras.",
    }

    if st.button("Verificar passo 2"):
        mostrar_feedback_codigo(codigo_2, requisitos_2)

    with st.expander("Ver solução possível"):
        st.code(
            'X = df[["idade", "altura_cm", "peso_kg", "imc"]]\n'
            'y = df["sexo"]',
            language="python",
        )

# =============================
# Step 3
# =============================
with tabs[2]:
    st.subheader("3. Dividir treino/teste")

    st.write(
        "Objetivo: separar os dados em treino e teste para avaliar a capacidade de generalização."
    )

    codigo_3 = st.text_area(
        "Escreve o código para dividir os dados em treino e teste:",
        value=(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=0.30,\n"
            "    random_state=42,\n"
            "    stratify=y\n"
            ")"
        ),
        height=200,
        key="codigo_3",
    )

    requisitos_3 = {
        "train_test_split": "Usa `train_test_split` para separar treino e teste.",
        "X_train": "Cria `X_train`.",
        "X_test": "Cria `X_test`.",
        "y_train": "Cria `y_train`.",
        "y_test": "Cria `y_test`.",
        "test_size": "Define a proporção de dados para teste com `test_size`.",
        "random_state": "Usa `random_state` para tornar os resultados reprodutíveis.",
        "stratify": "Usa `stratify=y` para manter a proporção das classes.",
    }

    if st.button("Verificar passo 3"):
        mostrar_feedback_codigo(codigo_3, requisitos_3)

    with st.expander("Ver solução possível"):
        st.code(
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y,\n"
            "    test_size=0.30,\n"
            "    random_state=42,\n"
            "    stratify=y\n"
            ")",
            language="python",
        )

# =============================
# Step 4
# =============================
with tabs[3]:
    st.subheader("4. Treinar modelo")

    st.write(
        "Objetivo: criar e treinar um classificador. Para começar, usa uma árvore de decisão."
    )

    codigo_4 = st.text_area(
        "Escreve o código para criar e treinar o modelo:",
        value=(
            "modelo = DecisionTreeClassifier(random_state=42)\n"
            "modelo.fit(X_train, y_train)"
        ),
        height=140,
        key="codigo_4",
    )

    requisitos_4 = {
        "DecisionTreeClassifier": "Cria o modelo com `DecisionTreeClassifier`.",
        "modelo": "Guarda o classificador numa variável chamada `modelo`.",
        ".fit": "Treina o modelo com `.fit(X_train, y_train)`.",
        "X_train": "Usa `X_train` no treino.",
        "y_train": "Usa `y_train` no treino.",
    }

    if st.button("Verificar passo 4"):
        mostrar_feedback_codigo(codigo_4, requisitos_4)

    with st.expander("Ver solução possível"):
        st.code(
            "modelo = DecisionTreeClassifier(random_state=42)\n"
            "modelo.fit(X_train, y_train)",
            language="python",
        )

# =============================
# Step 5
# =============================
with tabs[4]:
    st.subheader("5. Avaliar métricas")

    st.write(
        "Objetivo: prever no conjunto de teste e avaliar o desempenho com accuracy, "
        "matriz de confusão e relatório de classificação."
    )

    codigo_5 = st.text_area(
        "Escreve o código para avaliar o modelo:",
        value=(
            "y_pred = modelo.predict(X_test)\n"
            "accuracy_score(y_test, y_pred)\n"
            "confusion_matrix(y_test, y_pred)\n"
            "classification_report(y_test, y_pred)"
        ),
        height=180,
        key="codigo_5",
    )

    requisitos_5 = {
        "predict": "Usa `modelo.predict(X_test)` para obter previsões.",
        "y_pred": "Guarda as previsões numa variável chamada `y_pred`.",
        "accuracy_score": "Calcula a accuracy com `accuracy_score`.",
        "confusion_matrix": "Calcula a matriz de confusão com `confusion_matrix`.",
        "classification_report": "Usa `classification_report` para ver precision, recall e F1-score.",
    }

    if st.button("Verificar passo 5"):
        mostrar_feedback_codigo(codigo_5, requisitos_5)

    st.write("Depois de escreveres e verificares os passos, podes correr a pipeline controlada:")

    modelo_nome = st.selectbox(
        "Modelo a usar na execução controlada",
        ["Árvore de decisão", "Random forest", "Regressão logística"],
    )

    if st.button("Executar pipeline controlada"):
        acc, cm, report = executar_pipeline_supervisionado(
            df,
            modelo_nome=modelo_nome,
        )

        st.success("Pipeline executada com sucesso.")
        st.write(f"Accuracy: **{acc:.3f}**")

        st.write("Matriz de confusão")
        cm_df = pd.DataFrame(
            cm,
            index=["Real: Feminino", "Real: Masculino"],
            columns=["Previsto: Feminino", "Previsto: Masculino"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.write("Relatório de classificação")
        st.text(report)

    with st.expander("Ver solução possível"):
        st.code(
            "y_pred = modelo.predict(X_test)\n"
            "accuracy_score(y_test, y_pred)\n"
            "confusion_matrix(y_test, y_pred)\n"
            "classification_report(y_test, y_pred)",
            language="python",
        )

# =============================
# Step 6
# =============================
with tabs[5]:
    st.subheader("6. Interpretar")

    st.write(
        "Objetivo: interpretar os resultados com espírito crítico. "
        "Um modelo pode ter bons resultados técnicos e ainda assim ter limitações."
    )

    resposta = st.text_area(
        "Escreve uma breve interpretação dos resultados. Refere pelo menos uma limitação.",
        height=180,
        key="interpretacao",
    )

    if st.button("Verificar interpretação"):
        resposta_lower = resposta.lower()

        pontos = {
            "treino": "Boa: referiste treino/teste ou separação dos dados.",
            "teste": "Boa: referiste treino/teste ou avaliação em dados não vistos.",
            "overfitting": "Boa: identificaste o risco de overfitting.",
            "data leakage": "Boa: identificaste o risco de data leakage.",
            "enviesamento": "Boa: identificaste o risco de enviesamento.",
            "validação": "Boa: referiste a importância da validação.",
            "accuracy": "Boa: referiste a accuracy.",
            "classes": "Boa: referiste a distribuição das classes.",
            "sintético": "Boa: lembraste que parte do dataset é sintética.",
        }

        encontrou = False

        for palavra, feedback in pontos.items():
            if palavra in resposta_lower:
                st.success(feedback)
                encontrou = True

        if not encontrou:
            st.warning(
                "A interpretação ainda está muito genérica. Tenta referir pelo menos uma limitação, "
                "por exemplo: dados sintéticos, overfitting, data leakage, enviesamento, validação "
                "ou classes desequilibradas."
            )

        st.info(
            "Uma boa interpretação deve responder a três perguntas: "
            "o que o modelo parece fazer bem, onde pode falhar, "
            "e que validação adicional seria necessária."
        )

    with st.expander("Exemplo de interpretação possível"):
        st.write(
            "O modelo parece conseguir distinguir parcialmente os grupos com base em idade, altura, peso e IMC. "
            "No entanto, os resultados devem ser interpretados com cautela, porque parte dos dados é sintética "
            "e pode não representar a população real. Além disso, a accuracy não deve ser analisada isoladamente; "
            "é importante observar a matriz de confusão, precision, recall e possíveis diferenças entre classes."
        )

st.info(
    "Nota: os dados sintéticos servem apenas para fins pedagógicos e teste de código. "
    "Não devem ser usados para inferência científica sobre a população portuguesa."
)
