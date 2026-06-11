# Exemplo não supervisionado: K-means e PCA

Aplicação pedagógica em **Streamlit** para apoiar uma aula introdutória de aprendizagem não supervisionada em Python, no contexto de Ciência de Dados em Saúde.

A aplicação usa o **Breast Cancer Wisconsin Dataset**, incluído no `scikit-learn`, para mostrar como aplicar uma pipeline simples com:

- seleção de variáveis numéricas;
- normalização;
- K-means;
- PCA;
- interpretação de clusters;
- comparação posterior com o diagnóstico real.

O diagnóstico real **não é usado no treino do K-means**. É usado apenas depois, para discussão crítica.

---

## Objetivos pedagógicos

Esta aplicação foi criada para ajudar os estudantes a compreender que, em aprendizagem não supervisionada:

- não existe uma variável-alvo `y` usada no treino;
- o algoritmo procura padrões de semelhança entre observações;
- os resultados dependem das variáveis escolhidas;
- a normalização é essencial quando se usam distâncias;
- o número de clusters é uma decisão de análise;
- PCA pode ajudar a visualizar dados com muitas variáveis;
- clusters são agrupamentos matemáticos, não diagnósticos clínicos.

---

## Dataset

A aplicação usa o dataset `load_breast_cancer` do `scikit-learn`.

Este dataset contém medidas numéricas extraídas de imagens de massas mamárias. A variável de diagnóstico real está disponível no dataset, mas nesta aplicação é usada apenas para comparação posterior.

---

## Funcionalidades principais

A aplicação permite:

1. carregar automaticamente o dataset;
2. escolher variáveis numéricas para o clustering;
3. escolher o número de clusters `k`;
4. normalizar os dados com `StandardScaler`;
5. aplicar K-means;
6. calcular o silhouette score;
7. visualizar clusters com PCA;
8. observar o método do cotovelo;
9. comparar os clusters com o diagnóstico real;
10. discutir a interpretação e as limitações dos clusters.

---

## Requisitos

A aplicação usa as seguintes bibliotecas:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
```

---

## Instalação

Recomenda-se criar um ambiente virtual.

### 1. Criar ambiente virtual

```bash
python -m venv venv
```

### 2. Ativar o ambiente virtual

No Windows:

```bash
venv\Scripts\activate
```

No macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Como executar

Na pasta onde se encontra o ficheiro da aplicação, executar:

```bash
streamlit run streamlit_app.py
```

---

## Estrutura dos ficheiros

```text
.
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Pipeline usada na aplicação

A pipeline principal segue estes passos:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = load_breast_cancer(as_frame=True)
df = data.data

X = df[variaveis]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

## Nota pedagógica e ética

Esta aplicação é apenas um exemplo didático. Os clusters produzidos pelo K-means são agrupamentos matemáticos e não devem ser interpretados automaticamente como categorias clínicas.

Mesmo quando os clusters parecem coincidir parcialmente com diagnósticos reais, isso não significa que o modelo tenha descoberto uma entidade clínica válida. A interpretação exige conhecimento clínico, validação externa e análise crítica.

---

## Limitações

Esta aplicação é deliberadamente simples. Algumas limitações importantes:

- usa um dataset limpo e preparado;
- não aborda missing values;
- não faz validação externa;
- não testa estabilidade dos clusters;
- não compara vários algoritmos de clustering;
- usa PCA apenas como ferramenta de visualização;
- silhouette score é uma métrica técnica, não uma prova de relevância clínica.

---

## Mensagem principal

> Clusters são pistas, não diagnósticos.

A aprendizagem não supervisionada pode ser útil para explorar padrões, mas os resultados devem ser interpretados com prudência e validados antes de qualquer utilização em contexto clínico.
