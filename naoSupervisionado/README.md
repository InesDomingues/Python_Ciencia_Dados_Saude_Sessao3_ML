# Exemplo não supervisionado: K-means e PCA

Esta aplicação Streamlit apresenta um exemplo pedagógico de aprendizagem não supervisionada, usando K-means e PCA. A app foi pensada para apoiar uma aula ou formação introdutória sobre clustering, com uma componente visual e uma componente prática em formato de desafio.

## Objetivo

O objetivo principal é mostrar como um algoritmo de clustering pode agrupar observações semelhantes sem usar uma variável-alvo `y`.

A aplicação usa o **Breast Cancer Wisconsin Dataset**, incluído no `scikit-learn`. O diagnóstico real existe no dataset, mas não é usado pelo K-means para criar os clusters. Surge apenas posteriormente, para comparação crítica dos resultados.

## Funcionalidades principais

A aplicação inclui:

- descrição inicial do dataset;
- demonstração iterativa do K-means em duas dimensões;
- visualização dos centróides e da sua deslocação;
- desafio prático com passos guiados;
- caixas para os estudantes escreverem código;
- verificação automática simples dos elementos esperados no código;
- dicas e possíveis soluções para cada passo;
- aplicação de K-means;
- visualização dos clusters com PCA;
- comparação opcional com o diagnóstico real;
- cálculo do silhouette score;
- cálculo do Adjusted Rand Index.

## Estrutura recomendada dos ficheiros

```text
.
├── streamlit_app.py
├── requirements.txt
└── README.md
```

Se o ficheiro principal tiver outro nome, substitua `streamlit_app.py` pelo nome correto no comando de execução.

## Instalação

Recomenda-se a criação de um ambiente virtual.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Executar a aplicação

```bash
streamlit run streamlit_app.py
```

Depois de executar o comando, o Streamlit deverá abrir automaticamente a aplicação no navegador. Se isso não acontecer, copie o endereço indicado no terminal.

## Dependências principais

A aplicação usa:

- `streamlit` para a interface interativa;
- `pandas` para manipulação de dados;
- `numpy` para operações numéricas;
- `matplotlib` para visualização gráfica;
- `scikit-learn` para o dataset, normalização, K-means, PCA e métricas.

## Notas pedagógicas

Esta app foi desenhada para distinguir claramente aprendizagem supervisionada e não supervisionada.

No exemplo não supervisionado:

- não existe variável-alvo `y` durante o treino;
- o K-means agrupa observações com base em semelhanças entre variáveis;
- a normalização é importante porque o K-means usa distâncias;
- a PCA é usada apenas para visualização em duas dimensões;
- os clusters devem ser interpretados com cautela.

## Aviso importante

Os clusters encontrados pelo K-means são agrupamentos matemáticos. Não devem ser interpretados como diagnósticos clínicos.

O diagnóstico real do dataset é usado apenas para reflexão e comparação posterior, nunca para treinar o algoritmo de clustering.

## Referências científicas úteis

- Lloyd, Stuart P. (1982). “Least Squares Quantization in PCM”. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- MacQueen, James. (1967). “Some Methods for Classification and Analysis of Multivariate Observations”. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
- Rousseeuw, Peter J. (1987). “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis”. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Hubert, Lawrence, e Phipps Arabie. (1985). “Comparing Partitions”. *Journal of Classification*, 2, 193–218.
