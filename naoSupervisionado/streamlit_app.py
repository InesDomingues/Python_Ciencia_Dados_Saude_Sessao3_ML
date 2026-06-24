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



def preparar_dados_demo_kmeans(
    X: pd.DataFrame,
    variaveis_2d: list[str],
    max_amostras: int = 250,
    seed: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Prepara uma versão 2D dos dados para a demonstração iterativa do K-means.

    A visualização é feita nas unidades originais, mas as distâncias são calculadas
    nas variáveis normalizadas, tal como na pipeline principal.
    """
    df_demo = X[variaveis_2d].dropna().copy()

    if len(df_demo) > max_amostras:
        df_demo = df_demo.sample(n=max_amostras, random_state=seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_demo)

    return df_demo, X_scaled, scaler


def inicializar_demo_kmeans(
    X: pd.DataFrame,
    variaveis_2d: list[str],
    k: int,
    seed: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Inicializa centróides aleatórios a partir de observações existentes."""
    df_demo, X_scaled, scaler = preparar_dados_demo_kmeans(
        X=X,
        variaveis_2d=variaveis_2d,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    indices_centroides = rng.choice(
        X_scaled.shape[0],
        size=k,
        replace=False,
    )

    estado = {
        "df_demo": df_demo,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "variaveis_2d": variaveis_2d,
        "k": k,
        "centroides": X_scaled[indices_centroides].copy(),
        "centroides_anteriores": None,
        "labels": np.full(X_scaled.shape[0], -1, dtype=int),
        "iteracao": 0,
        "subpasso": "atribuir",
        "fase": "Centróides iniciais definidos. Falta atribuir os pontos.",
        "inercia": None,
        "historico_inercia": [],
        "ultimo_deslocamento": None,
        "convergiu": False,
        "assinatura": (tuple(variaveis_2d), k),
        "seed": seed,
    }

    return estado


def atribuir_pontos_demo_kmeans(estado: dict) -> dict:
    """Atribui cada ponto ao centróide mais próximo."""
    X_scaled = estado["X_scaled"]
    centroides = estado["centroides"]

    distancias_quadradas = np.sum(
        (X_scaled[:, None, :] - centroides[None, :, :]) ** 2,
        axis=2,
    )

    labels = np.argmin(distancias_quadradas, axis=1)
    inercia = float(np.sum(np.min(distancias_quadradas, axis=1)))

    estado["labels"] = labels
    estado["inercia"] = inercia
    estado["historico_inercia"].append(inercia)
    estado["subpasso"] = "atualizar"
    estado["fase"] = (
        "Atribuição concluída: cada observação foi associada ao centróide mais próximo. "
        "O próximo passo é recalcular os centróides."
    )

    return estado


def atualizar_centroides_demo_kmeans(estado: dict, tol: float = 1e-4) -> dict:
    """Move cada centróide para a média dos pontos que lhe foram atribuídos."""
    X_scaled = estado["X_scaled"]
    labels = estado["labels"]
    centroides_antigos = estado["centroides"].copy()
    centroides_novos = centroides_antigos.copy()

    for cluster in range(estado["k"]):
        mascara = labels == cluster

        if np.any(mascara):
            centroides_novos[cluster] = X_scaled[mascara].mean(axis=0)
        else:
            # Se um cluster ficar vazio, reposiciona-o no ponto mais afastado
            # do seu centróide mais próximo. Isto evita que a demonstração pare.
            distancias = np.sum(
                (X_scaled[:, None, :] - centroides_antigos[None, :, :]) ** 2,
                axis=2,
            )
            ponto_mais_afastado = int(np.argmax(np.min(distancias, axis=1)))
            centroides_novos[cluster] = X_scaled[ponto_mais_afastado]

    deslocamentos = np.linalg.norm(
        centroides_novos - centroides_antigos,
        axis=1,
    )

    # Inércia após mover os centróides, mantendo a atribuição atual.
    inercia = 0.0
    for cluster in range(estado["k"]):
        mascara = labels == cluster
        if np.any(mascara):
            inercia += float(
                np.sum((X_scaled[mascara] - centroides_novos[cluster]) ** 2)
            )

    estado["centroides_anteriores"] = centroides_antigos
    estado["centroides"] = centroides_novos
    estado["inercia"] = inercia
    estado["historico_inercia"].append(inercia)
    estado["ultimo_deslocamento"] = float(np.max(deslocamentos))
    estado["iteracao"] += 1
    estado["subpasso"] = "atribuir"

    if estado["ultimo_deslocamento"] < tol:
        estado["convergiu"] = True
        estado["fase"] = (
            "Os centróides praticamente deixaram de se mover. "
            "O algoritmo pode ser considerado convergido para esta inicialização."
        )
    else:
        estado["fase"] = (
            "Atualização concluída: os centróides foram movidos para a média dos pontos "
            "de cada cluster. O próximo passo é voltar a atribuir os pontos."
        )

    return estado


def avancar_passo_demo_kmeans(estado: dict) -> dict:
    """Executa o próximo subpasso do algoritmo."""
    if estado["convergiu"]:
        return estado

    if estado["subpasso"] == "atribuir":
        return atribuir_pontos_demo_kmeans(estado)

    return atualizar_centroides_demo_kmeans(estado)


def criar_figura_demo_kmeans(estado: dict) -> plt.Figure:
    """Desenha os pontos, os centróides atuais e o movimento dos centróides."""
    df_demo = estado["df_demo"]
    X_raw = df_demo[estado["variaveis_2d"]].to_numpy()
    labels = estado["labels"]

    centroides_raw = estado["scaler"].inverse_transform(estado["centroides"])
    centroides_anteriores = estado.get("centroides_anteriores")

    if centroides_anteriores is not None:
        centroides_anteriores_raw = estado["scaler"].inverse_transform(
            centroides_anteriores
        )
    else:
        centroides_anteriores_raw = None

    fig, ax = plt.subplots(figsize=(6.5, 5))

    cmap = plt.get_cmap("tab10")

    if np.all(labels == -1):
        ax.scatter(
            X_raw[:, 0],
            X_raw[:, 1],
            c="lightgray",
            s=28,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.25,
            label="Observações ainda sem cluster",
        )
    else:
        for cluster in range(estado["k"]):
            mascara = labels == cluster
            if np.any(mascara):
                ax.scatter(
                    X_raw[mascara, 0],
                    X_raw[mascara, 1],
                    color=cmap(cluster % 10),
                    s=30,
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=0.25,
                    label=f"Cluster {cluster}",
                )

    if centroides_anteriores_raw is not None:
        for cluster in range(estado["k"]):
            ax.annotate(
                "",
                xy=centroides_raw[cluster],
                xytext=centroides_anteriores_raw[cluster],
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                ),
            )
            ax.scatter(
                centroides_anteriores_raw[cluster, 0],
                centroides_anteriores_raw[cluster, 1],
                marker="x",
                color="black",
                s=85,
                linewidths=2,
            )

    for cluster in range(estado["k"]):
        ax.scatter(
            centroides_raw[cluster, 0],
            centroides_raw[cluster, 1],
            marker="X",
            color=cmap(cluster % 10),
            s=260,
            edgecolors="black",
            linewidths=1.2,
            label=f"Centróide {cluster}",
        )
        ax.text(
            centroides_raw[cluster, 0],
            centroides_raw[cluster, 1],
            str(cluster),
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color="white",
        )

    margem_x = 0.05 * (X_raw[:, 0].max() - X_raw[:, 0].min())
    margem_y = 0.05 * (X_raw[:, 1].max() - X_raw[:, 1].min())

    ax.set_xlim(X_raw[:, 0].min() - margem_x, X_raw[:, 0].max() + margem_x)
    ax.set_ylim(X_raw[:, 1].min() - margem_y, X_raw[:, 1].max() + margem_y)

    ax.set_xlabel(estado["variaveis_2d"][0])
    ax.set_ylabel(estado["variaveis_2d"][1])
    ax.set_title(
        f"iteração = {estado['iteracao']} | "
        f"próximo passo = {estado['subpasso']}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")

    return fig


def criar_figura_inercia_demo_kmeans(estado: dict) -> plt.Figure:
    """Desenha a evolução da inércia ao longo dos subpassos."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    historico = estado["historico_inercia"]

    if len(historico) > 0:
        ax.plot(range(1, len(historico) + 1), historico, marker="o")
        ax.set_xticks(range(1, len(historico) + 1))
    else:
        ax.text(
            0.5,
            0.5,
            "A inércia aparece depois da primeira atribuição.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Subpasso")
    ax.set_ylabel("Inércia")
    ax.grid(True, alpha=0.3)

    return fig


def avaliar_codigo(codigo: str, requisitos: dict) -> tuple[bool, list[str], list[str]]:
    """Avalia se o código do aluno contém elementos esperados.

    Nota pedagógica: esta função não executa o código livre do aluno.
    Apenas verifica se aparecem componentes essenciais do passo.
    """
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
    if not codigo.strip() or codigo.strip() == "Escreva aqui o seu código":
        st.warning("Escreva algum código antes de verificar.")
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


# =============================
# App
# =============================

st.title("Exemplo não supervisionado: K-means e PCA")

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
# Introdução
# =============================
st.header("Descrição dos dados")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Observações", len(X))

with col_b:
    st.metric("Variáveis disponíveis", X.shape[1])

st.write(
    "Neste exemplo, é usado o Breast Cancer Wisconsin Dataset incluído no scikit-learn. "
    "O dataset tem várias medidas numéricas calculadas a partir de imagens de massas mamárias."
)

st.info(
    "O diagnóstico real existe no dataset, mas não é usado pelo K-means. "
    "Pode ser mostrado apenas para avaliação crítica dos clusters obtidos."
)

with st.expander("Ver primeiras linhas dos dados"):
    st.dataframe(X.head(), use_container_width=True)

# =============================
# Configuração interna da análise
# =============================
# A antiga secção visual "Configuração da análise" foi removida.
# No entanto, a app continua a precisar destas variáveis para executar a pipeline.
variaveis_disponiveis = list(X.columns)

variaveis = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
]
variaveis = [v for v in variaveis if v in variaveis_disponiveis]

k = DEFAULT_K
mostrar_diagnostico_real = True

if len(variaveis) < 2:
    st.error("Não há variáveis suficientes para aplicar K-means e PCA.")
    st.stop()

resultados = preparar_resultados(
    X=X,
    diagnostico_real=diagnostico_real,
    variaveis=variaveis,
    k=k,
)

df_resultados = resultados["df_resultados"]

# =============================
# Demonstração iterativa do K-means
# =============================
st.header("Intuição: como é que o K-means aprende?")

st.write(
    "Nesta demonstração, o K-means é mostrado em duas dimensões para que seja possível "
    "ver os centróides a deslocarem-se. O algoritmo alterna entre dois subpassos: "
    "atribuir cada ponto ao centróide mais próximo e atualizar cada centróide para a média "
    "dos pontos que lhe foram atribuídos."
)

st.caption(
    "Nota: as distâncias são calculadas depois da normalização."
)

col_demo_var1, col_demo_var2, col_demo_k = st.columns([1, 1, 1])

indice_var1_default = (
    variaveis_disponiveis.index(variaveis[0])
    if len(variaveis) >= 1 and variaveis[0] in variaveis_disponiveis
    else 0
)

indice_var2_default = (
    variaveis_disponiveis.index(variaveis[1])
    if len(variaveis) >= 2 and variaveis[1] in variaveis_disponiveis
    else min(1, len(variaveis_disponiveis) - 1)
)

with col_demo_var1:
    variavel_x_demo = st.selectbox(
        "Variável no eixo X",
        options=variaveis_disponiveis,
        index=indice_var1_default,
        key="variavel_x_demo_kmeans",
    )

with col_demo_var2:
    variavel_y_demo = st.selectbox(
        "Variável no eixo Y",
        options=variaveis_disponiveis,
        index=indice_var2_default,
        key="variavel_y_demo_kmeans",
    )

with col_demo_k:
    k_demo = st.slider(
        "valor de k (número de clusters)",
        min_value=2,
        max_value=6,
        value=min(k, 6),
        key="k_demo_iterativo",
    )

if variavel_x_demo == variavel_y_demo:
    st.error("Escolha duas variáveis diferentes para a demonstração 2D.")
else:
    variaveis_demo = [variavel_x_demo, variavel_y_demo]
    assinatura_demo = (tuple(variaveis_demo), k_demo)

    if (
        "demo_kmeans_iterativo" not in st.session_state
        or st.session_state.demo_kmeans_iterativo.get("assinatura") != assinatura_demo
    ):
        st.session_state.demo_kmeans_iterativo = inicializar_demo_kmeans(
            X=X,
            variaveis_2d=variaveis_demo,
            k=k_demo,
            seed=DEFAULT_RANDOM_STATE,
        )

    estado_kmeans = st.session_state.demo_kmeans_iterativo

    col_fig_kmeans, col_info_kmeans = st.columns([1.35, 1])

    with col_fig_kmeans:
        fig_kmeans_demo = criar_figura_demo_kmeans(estado_kmeans)
        st.pyplot(fig_kmeans_demo, use_container_width=False)

        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

        with col_btn1:
            if st.button("Avançar 1 passo", key="btn_kmeans_avancar"):
                st.session_state.demo_kmeans_iterativo = avancar_passo_demo_kmeans(
                    st.session_state.demo_kmeans_iterativo
                )
                st.rerun()

        with col_btn2:
            if st.button("Completar ciclo", key="btn_kmeans_ciclo"):
                for _ in range(2):
                    st.session_state.demo_kmeans_iterativo = avancar_passo_demo_kmeans(
                        st.session_state.demo_kmeans_iterativo
                    )
                st.rerun()

        with col_btn3:
            if st.button("+5 passos", key="btn_kmeans_5_passos"):
                for _ in range(5):
                    st.session_state.demo_kmeans_iterativo = avancar_passo_demo_kmeans(
                        st.session_state.demo_kmeans_iterativo
                    )
                st.rerun()

        with col_btn4:
            if st.button("Reiniciar", key="btn_kmeans_reiniciar"):
                nova_seed = int(np.random.randint(0, 100000))
                st.session_state.demo_kmeans_iterativo = inicializar_demo_kmeans(
                    X=X,
                    variaveis_2d=variaveis_demo,
                    k=k_demo,
                    seed=nova_seed,
                )
                st.rerun()

    with col_info_kmeans:
        st.subheader("Estado do algoritmo")

        metrica_1, metrica_2 = st.columns(2)
        metrica_1.metric("Iteração completa", estado_kmeans["iteracao"])
        metrica_2.metric("Próximo subpasso", estado_kmeans["subpasso"])

        if estado_kmeans["inercia"] is None:
            st.metric("Inércia", "—")
        else:
            st.metric("Inércia", f"{estado_kmeans['inercia']:.2f}")

        if estado_kmeans["convergiu"]:
            st.success("Convergência atingida para esta inicialização.")
        else:
            st.info(estado_kmeans["fase"])

        fig_inercia_demo = criar_figura_inercia_demo_kmeans(estado_kmeans)
        st.pyplot(fig_inercia_demo, use_container_width=False)

with st.expander("Ver pseudocódigo do K-means iterativo"):
    st.code(
        """1. Escolher k centróides iniciais
2. Repetir até convergir:
      a) Atribuir cada observação ao centróide mais próximo
      b) Atualizar cada centróide para a média das observações do seu cluster
3. Parar quando os centróides deixam de se mover ou quando se atinge o limite de iterações""",
        language="text",
    )

# =============================
# Pipeline
# =============================
st.header("Desafio: exemplo prático não-supervisionado")

st.info(
    "Por segurança, a app não executa diretamente código livre. Primeiro verifica se o código "
    "contém os elementos essenciais. Quando os passos principais estão corretos, a app mostra "
    "uma versão controlada da pipeline."
)

tabs = st.tabs(
    [
        "0. Importar toolboxes",
        "1. Ler dados",
        "2. Escolher variáveis",
        "3. Normalizar dados",
        "4. Aplicar K-means",
        "5. Visualizar com PCA",
        "6. Interpretar clusters",
        "7. Correr a pipeline",
    ]
)

# =============================
# Step 0
# =============================
with tabs[0]:
    st.subheader("0. Importar toolboxes")

    st.write(
        "Objetivo: importar as bibliotecas e funções necessárias para carregar dados, "
        "normalizar variáveis, aplicar K-means, reduzir a dimensionalidade com PCA "
        "e avaliar os clusters."
    )

    codigo_0 = st.text_area(
        "Escreva o código para fazer os imports necessários:",
        value="Escreva aqui o seu código",
        height=240,
        key="codigo_0_nao_supervisionado",
    )

    requisitos_0 = {
        "pandas": "Importa o `pandas`, habitualmente como `pd`.",
        "load_breast_cancer": "Importa `load_breast_cancer` a partir de `sklearn.datasets`.",
        "StandardScaler": "Importa `StandardScaler` a partir de `sklearn.preprocessing`.",
        "KMeans": "Importa `KMeans` a partir de `sklearn.cluster`.",
        "PCA": "Importa `PCA` a partir de `sklearn.decomposition`.",
        "silhouette_score": "Importa `silhouette_score` a partir de `sklearn.metrics`.",
    }

    if st.button("Verificar passo 0", key="verificar_ns_0"):
        mostrar_feedback_codigo(codigo_0, requisitos_0)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Vai precisar de importar:

            - `pandas`, para organizar os dados;
            - `load_breast_cancer`, para carregar o dataset;
            - `StandardScaler`, para normalizar as variáveis;
            - `KMeans`, para criar os clusters;
            - `PCA`, para visualizar dados com muitas variáveis em duas dimensões;
            - `silhouette_score`, para uma avaliação técnica simples dos clusters.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "import pandas as pd\n"
            "\n"
            "from sklearn.datasets import load_breast_cancer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.metrics import silhouette_score",
            language="python",
        )

# =============================
# Step 1
# =============================
with tabs[1]:
    st.subheader("1. Ler dados")

    st.write(
        "Objetivo: carregar o Breast Cancer Wisconsin Dataset e guardar as variáveis "
        "numéricas num DataFrame chamado `df`."
    )

    codigo_1 = st.text_area(
        "Escreva o código para carregar os dados:",
        value="Escreva aqui o seu código",
        height=180,
        key="codigo_1_nao_supervisionado",
    )

    requisitos_1 = {
        "load_breast_cancer": "Usa `load_breast_cancer(...)` para carregar o dataset.",
        "as_frame": "Usa `as_frame=True` para obter os dados num formato compatível com pandas.",
        "df": "Guarda as variáveis numéricas numa variável chamada `df`.",
        "data.data": "Usa `data.data` para obter as variáveis preditoras.",
    }

    if st.button("Verificar passo 1", key="verificar_ns_1"):
        mostrar_feedback_codigo(codigo_1, requisitos_1)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Primeiro carregue o dataset com `load_breast_cancer(as_frame=True)`.
            Depois, use o atributo `.data` para obter as variáveis numéricas.

            O diagnóstico real existe no dataset, mas não deve ser usado para treinar o K-means.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "data = load_breast_cancer(as_frame=True)\n"
            "df = data.data",
            language="python",
        )

# =============================
# Step 2
# =============================
with tabs[2]:
    st.subheader("2. Escolher variáveis")

    st.write(
        "Objetivo: escolher as variáveis numéricas que vão entrar no clustering. "
        "Ao contrário do exemplo supervisionado, aqui não é definida uma variável-alvo `y`."
    )

    codigo_2 = st.text_area(
        "Escreva o código para selecionar as variáveis:",
        value="Escreva aqui o seu código",
        height=220,
        key="codigo_2_nao_supervisionado",
    )

    requisitos_2 = {
        "X": "Cria uma variável chamada `X` com as variáveis usadas no clustering.",
        "mean radius": "Inclui a variável `mean radius`.",
        "mean texture": "Inclui a variável `mean texture`.",
        "mean perimeter": "Inclui a variável `mean perimeter`.",
        "mean area": "Inclui a variável `mean area`.",
        "mean smoothness": "Inclui a variável `mean smoothness`.",
        "mean compactness": "Inclui a variável `mean compactness`.",
    }

    if st.button("Verificar passo 2", key="verificar_ns_2"):
        mostrar_feedback_codigo(codigo_2, requisitos_2)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Selecione algumas colunas do DataFrame.

            Neste exemplo não é usado `y` para treinar o modelo, porque o K-means procura grupos apenas a partir das semelhanças entre observações.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "X = df[[\n"
            "    \"mean radius\",\n"
            "    \"mean texture\",\n"
            "    \"mean perimeter\",\n"
            "    \"mean area\",\n"
            "    \"mean smoothness\",\n"
            "    \"mean compactness\",\n"
            "]]",
            language="python",
        )

# =============================
# Step 3
# =============================
with tabs[3]:
    st.subheader("3. Normalizar dados")

    st.write(
        "Objetivo: colocar as variáveis numa escala comparável. Isto é especialmente importante "
        "no K-means, porque o algoritmo usa distâncias."
    )

    codigo_3 = st.text_area(
        "Escreva o código para normalizar os dados:",
        value="Escreva aqui o seu código",
        height=180,
        key="codigo_3_nao_supervisionado",
    )

    requisitos_3 = {
        "StandardScaler": "Cria um normalizador com `StandardScaler()`.",
        "scaler": "Guarda o normalizador numa variável chamada `scaler`.",
        "fit_transform": "Usa `fit_transform` para aprender a normalização e transformar `X`.",
        "X_scaled": "Guarda os dados normalizados numa variável chamada `X_scaled`.",
    }

    if st.button("Verificar passo 3", key="verificar_ns_3"):
        mostrar_feedback_codigo(codigo_3, requisitos_3)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Como o K-means trabalha com distâncias, uma variável com valores numericamente muito grandes pode dominar a formação dos clusters.

            Use `StandardScaler()` e aplique `fit_transform` às variáveis selecionadas.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)",
            language="python",
        )

# =============================
# Step 4
# =============================
with tabs[4]:
    st.subheader("4. Aplicar K-means")

    st.write(
        "Objetivo: criar um modelo K-means, definir o número de clusters e atribuir cada observação "
        "a um cluster."
    )

    codigo_4 = st.text_area(
        "Escreva o código para aplicar o K-means:",
        value="Escreva aqui o seu código",
        height=220,
        key="codigo_4_nao_supervisionado",
    )

    requisitos_4 = {
        "KMeans": "Cria o modelo com `KMeans`.",
        "n_clusters": "Define o número de clusters com `n_clusters`.",
        "random_state": "Define `random_state` para tornar o resultado reprodutível.",
        "fit_predict": "Usa `fit_predict` para treinar o modelo e obter os clusters.",
        "clusters": "Guarda os rótulos dos clusters numa variável chamada `clusters`.",
        "X_scaled": "Aplica o K-means aos dados normalizados `X_scaled`.",
    }

    if st.button("Verificar passo 4", key="verificar_ns_4"):
        mostrar_feedback_codigo(codigo_4, requisitos_4)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Crie um modelo com `KMeans(n_clusters=2, random_state=42, n_init=10)`.

            Depois use `.fit_predict(X_scaled)` para obter o cluster atribuído a cada observação.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "kmeans = KMeans(\n"
            "    n_clusters=2,\n"
            "    random_state=42,\n"
            "    n_init=10,\n"
            ")\n"
            "\n"
            "clusters = kmeans.fit_predict(X_scaled)",
            language="python",
        )

# =============================
# Step 5
# =============================
with tabs[5]:
    st.subheader("5. Visualizar com PCA")

    st.write(
        "Objetivo: reduzir os dados para duas componentes principais, para visualizar os clusters "
        "num gráfico 2D."
    )

    codigo_5 = st.text_area(
        "Escreva o código para aplicar PCA:",
        value="Escreva aqui o seu código",
        height=180,
        key="codigo_5_nao_supervisionado",
    )

    requisitos_5 = {
        "PCA": "Cria uma PCA com `PCA(n_components=2)`.",
        "n_components": "Define `n_components=2` para obter duas componentes principais.",
        "fit_transform": "Usa `fit_transform` para calcular a projeção PCA.",
        "X_pca": "Guarda o resultado numa variável chamada `X_pca`.",
        "X_scaled": "Aplica a PCA aos dados normalizados `X_scaled`.",
    }

    if st.button("Verificar passo 5", key="verificar_ns_5"):
        mostrar_feedback_codigo(codigo_5, requisitos_5)

    with st.expander("Ver dica"):
        st.markdown(
            """
            A PCA não cria os clusters. Serve aqui para visualizar, em 2D, dados que originalmente têm mais variáveis.

            Use `PCA(n_components=2)` e aplique-o aos dados normalizados.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "pca = PCA(n_components=2)\n"
            "X_pca = pca.fit_transform(X_scaled)",
            language="python",
        )

# =============================
# Step 6
# =============================
with tabs[6]:
    st.subheader("6. Interpretar clusters")

    st.write(
        "Objetivo: criar uma tabela com os clusters, observar as médias por cluster e calcular "
        "uma métrica técnica simples, como o silhouette score."
    )

    codigo_6 = st.text_area(
        "Escreva o código para interpretar os clusters:",
        value="Escreva aqui o seu código",
        height=240,
        key="codigo_6_nao_supervisionado",
    )

    requisitos_6 = {
        "cluster": "Cria uma coluna ou variável com os clusters atribuídos.",
        "X_pca": "Usa `X_pca` para guardar as coordenadas da projeção PCA.",
        "PC1": "Cria uma coluna `PC1` com a primeira componente principal.",
        "PC2": "Cria uma coluna `PC2` com a segunda componente principal.",
        "groupby": "Usa `groupby` para resumir as variáveis por cluster.",
        "mean": "Calcula médias por cluster com `.mean()`.",
        "silhouette_score": "Calcula o `silhouette_score`.",
        "X_scaled": "Usa os dados normalizados no cálculo do silhouette score.",
        "clusters": "Usa os clusters atribuídos pelo K-means.",
    }

    if st.button("Verificar passo 6", key="verificar_ns_6"):
        mostrar_feedback_codigo(codigo_6, requisitos_6)

    with st.expander("Ver dica"):
        st.markdown(
            """
            Junte os clusters ao DataFrame original para comparar as médias das variáveis em cada grupo.

            Acrescente também `PC1` e `PC2`, usando o resultado da PCA, para facilitar a visualização dos clusters no gráfico 2D.

            O `silhouette_score` pode ajudar a avaliar se os clusters estão separados, mas não prova que tenham significado clínico.
            """
        )

    with st.expander("Ver possível solução"):
        st.code(
            "df_resultados = X.copy()\n"
            "df_resultados[\"cluster\"] = clusters\n"
            "df_resultados[\"PC1\"] = X_pca[:, 0]\n"
            "df_resultados[\"PC2\"] = X_pca[:, 1]\n"
            "\n"
            "resumo_clusters = df_resultados.groupby(\"cluster\").mean()\n"
            "sil = silhouette_score(X_scaled, clusters)\n"
            "\n"
            "print(\"Resumo dos clusters:\")\n"
            "print(resumo_clusters)\n"
            "\n"
            "print(\"Silhouette score:\")\n"
            "print(sil)",
            language="python",
        )

# =============================
# Step 7
# =============================
with tabs[7]:
    st.subheader("7. Correr a pipeline controlada")

    st.write(
        "Objetivo: ver a pipeline completa a funcionar e interpretar os resultados com espírito crítico. "
        "Um bom resultado técnico não significa automaticamente que os clusters tenham significado clínico."
    )

    codigo_completo = f"""
# 0. Importar bibliotecas e funções
import pandas as pd

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

# Nota: aqui não se define y.
# O K-means não usa diagnóstico real para criar os clusters.

# 3. Normalizar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Aplicar K-means
kmeans = KMeans(
    n_clusters={k},
    random_state=42,
    n_init=10,
)
clusters = kmeans.fit_predict(X_scaled)

# 5. Visualizar com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Interpretar clusters
df_resultados = X.copy()
df_resultados["cluster"] = clusters
df_resultados["PC1"] = X_pca[:, 0]
df_resultados["PC2"] = X_pca[:, 1]

resumo_clusters = df_resultados.groupby("cluster").mean()
sil = silhouette_score(X_scaled, clusters)

print("Resumo dos clusters:")
print(resumo_clusters)

print("Silhouette score:")
print(sil)
"""

    with st.expander("Ver código completo da pipeline que será executada"):
        st.code(codigo_completo, language="python")

    if st.button("Executar pipeline", key="executar_pipeline_nao_supervisionada"):
        st.success("Pipeline executada com sucesso.")
        st.write(f"Silhouette score: **{resultados['silhouette']:.3f}**")

        st.write("Tamanho e médias das variáveis por cluster")
        st.dataframe(resultados["resumo_clusters"], use_container_width=True)

        st.write("Tamanho dos clusters")
        st.dataframe(
            df_resultados["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="n"),
            use_container_width=True,
        )

        if mostrar_diagnostico_real:
            st.write("Comparação com diagnóstico real")
            tabela = pd.crosstab(
                df_resultados["cluster"],
                df_resultados["diagnostico_real"],
            )
            st.dataframe(tabela, use_container_width=True)

            diagnostico_codificado = pd.Categorical(
                df_resultados["diagnostico_real"]
            ).codes
            ari = adjusted_rand_score(
                diagnostico_codificado,
                df_resultados["cluster"],
            )

        st.write("Visualização PCA")

        if mostrar_diagnostico_real:
            col_pca_clusters, col_pca_real = st.columns(2)

            with col_pca_clusters:
                fig_clusters = criar_figura_pca_clusters(df_resultados)
                st.pyplot(fig_clusters, use_container_width=True)

            with col_pca_real:
                fig_real = criar_figura_pca_diagnostico_real(df_resultados)
                st.pyplot(fig_real, use_container_width=True)
        else:
            fig_clusters = criar_figura_pca_clusters(df_resultados)
            st.pyplot(fig_clusters, use_container_width=True)

        st.info(
            "Para pensar: Os clusters são estáveis? As variáveis escolhidas fazem sentido? "
            "Os grupos têm interpretação clínica plausível?"
        )

# =============================
# Fecho
# =============================
st.header("Mensagem final")

st.write(
    "O clustering pode ajudar a explorar padrões, mas não prova que existam grupos clínicos reais. "
    "A interpretação depende das variáveis escolhidas, da escala, do número de clusters e da validação posterior."
)

st.error(
    "Clusters são pistas, não diagnósticos. "
    "Antes de usar qualquer agrupamento em saúde, é necessário validar os resultados com dados independentes e conhecimento clínico."
)
