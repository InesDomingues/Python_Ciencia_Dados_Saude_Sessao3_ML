from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# =============================
# Configuração
# =============================
DEFAULT_RANDOM_STATE = 42
DEFAULT_K = 2

st.set_page_config(
    page_title="Exemplo não supervisionado: K-means e PCA",
    page_icon="🧭",
    layout="wide",
)

# =============================
# Funções auxiliares
# =============================

@st.cache_data
def carregar_dataset_breast_cancer() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Carrega o Breast Cancer Wisconsin Dataset incluído no scikit-learn.

    O diagnóstico real é guardado apenas para comparação posterior.
    Não é usado pelo K-means.
    """
    data = load_breast_cancer(as_frame=True)

    df = data.data.copy()
    target = data.target.copy()
    target_names = list(data.target_names)

    diagnostico_real = target.map({i: nome for i, nome in enumerate(target_names)})

    return df, diagnostico_real, target_names


def normalizar_dados(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Normaliza as variáveis numéricas com média 0 e desvio-padrão 1."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def aplicar_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    """Treina K-means."""
    modelo = KMeans(
        n_clusters=k,
        random_state=DEFAULT_RANDOM_STATE,
        n_init=10,
    )
    modelo.fit(X_scaled)
    return modelo


def aplicar_pca(X_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    """Reduz os dados para duas componentes principais."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def criar_figura_pca_clusters(df_resultados: pd.DataFrame) -> plt.Figure:
    """Cria scatter plot PCA colorido por cluster."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for cluster in sorted(df_resultados["cluster"].unique()):
        dados_cluster = df_resultados[df_resultados["cluster"] == cluster]
        ax.scatter(
            dados_cluster["PC1"],
            dados_cluster["PC2"],
            label=f"Cluster {cluster}",
            alpha=0.75,
            s=35,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title("Visualização dos clusters após PCA")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig


def criar_figura_pca_diagnostico_real(df_resultados: pd.DataFrame) -> plt.Figure:
    """Cria scatter plot PCA colorido pelo diagnóstico real.

    Esta figura serve apenas para comparação posterior.
    O diagnóstico real não é usado pelo K-means.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for categoria in sorted(df_resultados["diagnostico_real"].unique()):
        dados_categoria = df_resultados[df_resultados["diagnostico_real"] == categoria]
        ax.scatter(
            dados_categoria["PC1"],
            dados_categoria["PC2"],
            label=categoria,
            alpha=0.75,
            s=35,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.set_title("Mesma projeção PCA, colorida pelo diagnóstico real")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return fig


def criar_figura_elbow(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> plt.Figure:
    """Cria gráfico do método do cotovelo."""
    valores_k = list(range(k_min, k_max + 1))
    inercias = []

    for k in valores_k:
        modelo = aplicar_kmeans(X_scaled, k)
        inercias.append(modelo.inertia_)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(valores_k, inercias, marker="o")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inércia")
    ax.set_title("Método do cotovelo")
    ax.grid(True, alpha=0.3)

    return fig


def calcular_resumo_clusters(
    X_original: pd.DataFrame,
    clusters: np.ndarray,
    variaveis: list[str],
) -> pd.DataFrame:
    """Calcula médias por cluster nas variáveis originais."""
    df_temp = X_original.copy()
    df_temp["cluster"] = clusters

    resumo = df_temp.groupby("cluster")[variaveis].mean().round(2)
    resumo.insert(0, "n", df_temp["cluster"].value_counts().sort_index())

    return resumo


def preparar_resultados(
    X: pd.DataFrame,
    diagnostico_real: pd.Series,
    variaveis: list[str],
    k: int,
) -> dict:
    """Executa a pipeline não supervisionada completa."""
    X_selecionado = X[variaveis].dropna().copy()
    diagnostico_alinhado = diagnostico_real.loc[X_selecionado.index]

    X_scaled, scaler = normalizar_dados(X_selecionado)

    kmeans = aplicar_kmeans(X_scaled, k)
    clusters = kmeans.labels_

    X_pca, pca = aplicar_pca(X_scaled)

    df_resultados = X_selecionado.copy()
    df_resultados["cluster"] = clusters
    df_resultados["PC1"] = X_pca[:, 0]
    df_resultados["PC2"] = X_pca[:, 1]
    df_resultados["diagnostico_real"] = diagnostico_alinhado.values

    sil = silhouette_score(X_scaled, clusters)

    return {
        "X_selecionado": X_selecionado,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "kmeans": kmeans,
        "clusters": clusters,
        "pca": pca,
        "X_pca": X_pca,
        "df_resultados": df_resultados,
        "silhouette": sil,
        "resumo_clusters": calcular_resumo_clusters(X_selecionado, clusters, variaveis),
    }


# =============================
# App
# =============================

st.title("🧭 Exemplo não supervisionado: K-means e PCA")

st.write(
    "Esta aplicação mostra uma pipeline simples de aprendizagem não supervisionada. "
    "O objetivo é agrupar observações semelhantes usando K-means e visualizar os resultados com PCA."
)

st.warning(
    "Importante: o K-means não usa o diagnóstico real. "
    "Os clusters são agrupamentos matemáticos, não diagnósticos clínicos."
)

X, diagnostico_real, target_names = carregar_dataset_breast_cancer()

# =============================
# Sidebar
# =============================
st.sidebar.header("Configuração")

variaveis_disponiveis = list(X.columns)

variaveis_default = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
]

variaveis_default = [v for v in variaveis_default if v in variaveis_disponiveis]

variaveis = st.sidebar.multiselect(
    "Variáveis usadas no clustering",
    options=variaveis_disponiveis,
    default=variaveis_default,
)

k = st.sidebar.slider(
    "Número de clusters (k)",
    min_value=2,
    max_value=8,
    value=DEFAULT_K,
)

mostrar_diagnostico_real = st.sidebar.checkbox(
    "Mostrar comparação com diagnóstico real",
    value=True,
)

if len(variaveis) < 2:
    st.error("Escolha pelo menos duas variáveis para aplicar K-means e PCA.")
    st.stop()

resultados = preparar_resultados(
    X=X,
    diagnostico_real=diagnostico_real,
    variaveis=variaveis,
    k=k,
)

df_resultados = resultados["df_resultados"]

# =============================
# Introdução
# =============================
st.header("1. O que vamos fazer?")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Observações", len(X))

with col_b:
    st.metric("Variáveis disponíveis", X.shape[1])

with col_c:
    st.metric("Variáveis usadas", len(variaveis))

st.write(
    "Neste exemplo, usamos o Breast Cancer Wisconsin Dataset incluído no scikit-learn. "
    "O dataset tem várias medidas numéricas calculadas a partir de imagens de massas mamárias. "
    "Para o exercício não supervisionado, usamos apenas as variáveis numéricas."
)

with st.expander("Ver primeiras linhas dos dados"):
    st.dataframe(X.head(), use_container_width=True)

# =============================
# Pipeline
# =============================
st.header("2. Pipeline não supervisionada")

tabs = st.tabs(
    [
        "0. Imports",
        "1. Escolher variáveis",
        "2. Normalizar",
        "3. K-means",
        "4. PCA",
        "5. Interpretar clusters",
        "6. Código completo",
    ]
)

with tabs[0]:
    st.subheader("0. Importar toolboxes")

    st.code(
        """import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score""",
        language="python",
    )

    st.write(
        "`pandas` organiza os dados, `StandardScaler` normaliza as variáveis, "
        "`KMeans` cria os clusters e `PCA` permite visualizar dados com muitas variáveis em 2D."
    )

with tabs[1]:
    st.subheader("1. Escolher variáveis numéricas")

    st.code(
        """data = load_breast_cancer(as_frame=True)

df = data.data

X = df[[
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
]]

# Aqui não existe y para treinar o modelo.
# Estamos num problema não supervisionado.""",
        language="python",
    )

    st.info(
        "No supervisionado temos uma variável-alvo `y`. "
        "Aqui não há `y` no treino: o algoritmo só vê as variáveis numéricas."
    )

    st.write("Variáveis atualmente selecionadas:")
    st.write(variaveis)

with tabs[2]:
    st.subheader("2. Normalizar dados")

    st.code(
        """scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)""",
        language="python",
    )

    st.write(
        "A normalização é essencial no K-means porque o algoritmo usa distâncias. "
        "Sem normalização, variáveis com valores maiores podem dominar os clusters."
    )

with tabs[3]:
    st.subheader("3. Aplicar K-means")

    st.code(
        f"""kmeans = KMeans(
    n_clusters={k},
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(X_scaled)""",
        language="python",
    )

    st.write(f"Número de clusters escolhido: **{k}**")
    st.write(f"Silhouette score: **{resultados['silhouette']:.3f}**")

    st.caption(
        "O silhouette score resume a separação entre clusters. "
        "Um valor mais alto sugere grupos mais separados, mas não garante relevância clínica."
    )

    fig_elbow = criar_figura_elbow(resultados["X_scaled"])
    st.pyplot(fig_elbow, use_container_width=False)

with tabs[4]:
    st.subheader("4. Visualizar com PCA")

    st.code(
        """pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)""",
        language="python",
    )

    explained = resultados["pca"].explained_variance_ratio_

    st.write(
        f"As duas primeiras componentes principais explicam "
        f"**{100 * explained.sum():.1f}%** da variabilidade das variáveis selecionadas."
    )

    fig_clusters = criar_figura_pca_clusters(df_resultados)
    st.pyplot(fig_clusters, use_container_width=False)

with tabs[5]:
    st.subheader("5. Interpretar clusters")

    st.write("Tamanho e médias das variáveis por cluster:")
    st.dataframe(resultados["resumo_clusters"], use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Tamanho dos clusters")
        st.dataframe(
            df_resultados["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="n"),
            use_container_width=True,
        )

    with col2:
        if mostrar_diagnostico_real:
            st.write("Comparação com diagnóstico real")
            tabela = pd.crosstab(
                df_resultados["cluster"],
                df_resultados["diagnostico_real"],
            )
            st.dataframe(tabela, use_container_width=True)

            diagnostico_codificado = pd.Categorical(df_resultados["diagnostico_real"]).codes
            ari = adjusted_rand_score(diagnostico_codificado, df_resultados["cluster"])
            st.metric("Adjusted Rand Index", f"{ari:.3f}")

    st.info(
        "Perguntas para discussão: Os clusters são estáveis? "
        "As variáveis escolhidas fazem sentido? "
        "Os grupos têm interpretação clínica plausível? "
        "Como poderíamos validar estes padrões?"
    )

    if mostrar_diagnostico_real:
        fig_real = criar_figura_pca_diagnostico_real(df_resultados)
        st.pyplot(fig_real, use_container_width=False)

with tabs[6]:
    st.subheader("6. Código completo da pipeline")

    codigo_completo = f"""
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Ler dados
data = load_breast_cancer(as_frame=True)
df = data.data

# 2. Escolher variáveis
X = df[{variaveis!r}]

# 3. Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Aplicar K-means
kmeans = KMeans(
    n_clusters={k},
    random_state=42,
    n_init=10,
)
clusters = kmeans.fit_predict(X_scaled)

# 5. Reduzir dimensionalidade para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Avaliação técnica simples
sil = silhouette_score(X_scaled, clusters)
print(sil)
"""
    st.code(codigo_completo, language="python")

# =============================
# Fecho
# =============================
st.header("3. Mensagem final")

st.write(
    "O clustering pode ajudar a explorar padrões, mas não prova que existam grupos clínicos reais. "
    "A interpretação depende das variáveis escolhidas, da escala, do número de clusters e da validação posterior."
)

st.error(
    "Clusters são pistas, não diagnósticos. "
    "Antes de usar qualquer agrupamento em saúde, é necessário validar os resultados com dados independentes e conhecimento clínico."
)
