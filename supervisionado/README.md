# Exemplo: Dataset de Altura e Peso

Aplicação pedagógica em **Streamlit** para apoiar uma aula introdutória de aprendizagem automática em Python, no contexto de Ciência de Dados em Saúde.

A aplicação permite recolher observações, combiná-las com um dataset sintético inicial, visualizar os dados e executar uma pipeline supervisionada. Inclui ainda uma demonstração visual de uma rede neuronal linear simples para mostrar, passo a passo, como um modelo ajusta pesos e altera a fronteira de decisão.

---

## Objetivos pedagógicos

Esta aplicação foi criada para ajudar a:

- compreender a estrutura de um dataset tabular;
- distinguir variáveis preditoras (`X`) e variável-alvo (`y`);
- visualizar relações entre variáveis, como altura e peso;
- perceber a diferença entre dados sintéticos e dados reais;
- treinar e avaliar modelos supervisionados simples;
- compreender a importância da divisão treino/teste;
- aplicar normalização de variáveis numéricas;
- interpretar métricas como accuracy, matriz de confusão e relatório de classificação;
- visualizar intuitivamente como uma rede neuronal linear aprende.

---

## Funcionalidades principais

A aplicação inclui quatro componentes principais.

### 1. Recolha de dados

É possível introduzir:

- idade;
- sexo;
- altura;
- peso.

A aplicação calcula automaticamente:

- IMC;
- categoria de IMC;
- origem da observação.

---

### 2. Visualização dos dados

A aplicação apresenta:

- resumo do número total de observações;
- número de observações sintéticas e reais;
- distribuição por sexo;
- scatter plot de altura vs. peso;
- linha de regressão linear.

Os dados sintéticos e reais são representados de forma diferente para facilitar a interpretação visual.

---

### 3. Demonstração de uma ANN linear

A aplicação inclui uma demonstração de uma rede neuronal muito simples:

- duas entradas: altura e peso;
- um bias;
- uma combinação linear;
- função sigmoid;
- saída interpretada como probabilidade de pertencer à classe “Masculino”.

A cada clique, o modelo realiza uma ou mais iterações de treino e atualiza:

- pesos;
- bias;
- loss;
- accuracy;
- fronteira de decisão.

Esta componente serve para mostrar que um modelo aprende através do ajuste de parâmetros.

---

### 4. Desafio supervisionado em Python

Pipeline supervisionada em vários passos:

0. importar bibliotecas;
1. ler dados;
2. definir `X` e `y`;
3. dividir treino/teste;
4. normalizar dados;
5. treinar modelo;
6. avaliar métricas;
7. correr a pipeline controlada.

Por segurança, a aplicação **não executa diretamente código livre**. Em vez disso, verifica se o código contém os elementos principais esperados e, no final, executa uma versão controlada da pipeline.

---

## Modelos disponíveis

Na pipeline controlada, é possível escolher entre:

- Árvore de decisão;
- Random forest;
- Regressão logística.

Os modelos são treinados para prever a variável `sexo` a partir de:

```python
altura_cm
peso_kg
```

Este exemplo é utilizado apenas para fins pedagógicos, por ser simples e visualmente intuitivo.

---

## Dataset

O dataset contém as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `id` | Identificador da observação |
| `idade` | Idade da pessoa |
| `sexo` | Sexo indicado no formulário |
| `altura_cm` | Altura em centímetros |
| `peso_kg` | Peso em quilogramas |
| `imc` | Índice de Massa Corporal |
| `categoria_imc` | Categoria de IMC |
| `origem` | Indica se a observação é sintética ou real |

---

## Nota pedagógica e ética

Este exemplo usa a previsão de `sexo` a partir de altura e peso apenas para fins didáticos, porque permite visualizar facilmente uma fronteira de decisão num espaço bidimensional.

Os dados sintéticos incluídos na aplicação servem apenas para teste e ensino. Não devem ser usados para inferência científica sobre a população portuguesa.

---

## Principais conceitos trabalhados

- Dados sintéticos;
- Recolha de dados em aula;
- Visualização exploratória;
- Regressão linear visual;
- Classificação supervisionada;
- Divisão treino/teste;
- Normalização com `StandardScaler`;
- Árvore de decisão;
- Random forest;
- Regressão logística;
- Accuracy;
- Matriz de confusão;
- Relatório de classificação;
- Rede neuronal linear;
- Sigmoid;
- Pesos, bias, loss e fronteira de decisão.

---

## Limitações

Esta aplicação é deliberadamente simples. Não pretende representar uma análise científica completa nem um sistema de decisão real.

Algumas limitações:

- o dataset sintético não representa a população portuguesa;
- a variável-alvo usada no exemplo tem finalidade apenas pedagógica;
- a avaliação é feita com uma divisão treino/teste simples;
- os modelos são usados com configurações básicas;
- a verificação automática do código é baseada na presença de palavras-chave, não numa avaliação semântica completa.
