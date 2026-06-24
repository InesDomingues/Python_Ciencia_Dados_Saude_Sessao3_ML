# Exemplo não supervisionado: K-means e PCA

Esta aplicação apresenta um exemplo pedagógico de aprendizagem não supervisionada, com K-means e PCA. 

## Objetivo

O objetivo principal é mostrar como um algoritmo de clustering agrupa observações semelhantes sem usar uma variável-alvo `y`.

A aplicação usa o **Breast Cancer Wisconsin Dataset**, incluído no `scikit-learn`. O diagnóstico real existe no dataset, mas não é usado pelo K-means para criar os clusters. Surge apenas posteriormente, para comparação dos resultados.

## Funcionalidades principais

A aplicação inclui:

- descrição inicial do dataset;
- demonstração iterativa do K-means em duas dimensões;
- visualização dos centróides e da sua deslocação;
- desafio prático com passos guiados;
- caixas para escrever código;
- verificação simples dos elementos esperados no código;
- dicas e possíveis soluções para cada passo;
- aplicação de K-means;
- visualização dos clusters com PCA;
- comparação com o diagnóstico real;
- cálculo do silhouette score;

## Notas 

No exemplo não supervisionado:

- não existe variável-alvo `y` durante o treino;
- o K-means agrupa observações com base em semelhanças entre variáveis;
- a normalização é importante porque o K-means usa distâncias;
- a PCA é usada apenas para visualização em duas dimensões;
- os clusters devem ser interpretados com cautela.

## Aviso

Os clusters encontrados pelo K-means são agrupamentos matemáticos. Não devem ser interpretados como diagnósticos clínicos.

O diagnóstico real do dataset é usado apenas para reflexão e comparação posterior, nunca para treinar o algoritmo de clustering.
