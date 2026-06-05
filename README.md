# Streamlit App — Dataset de Altura e Peso

Esta aplicação em Streamlit permite criar um dataset sintético inicial de adultos e adicionar novas observações introduzidas por estudantes.

A app permite:

- gerar um dataset sintético inicial;
- recolher idade, sexo, altura e peso;
- calcular automaticamente o IMC;
- classificar a categoria de IMC;
- distinguir dados sintéticos de dados introduzidos pelos alunos;
- visualizar um scatter plot de altura vs peso, com sexo por cor e origem por marcador;
- descarregar o dataset actualizado em CSV.

## Ficheiros

- `streamlit_app.py` — aplicação principal em Streamlit;
- `requirements.txt` — dependências necessárias;
- `README.md` — instruções de utilização.

## Instalação

Crie um ambiente virtual, se desejar:

```bash
python -m venv .venv
```

Active o ambiente virtual.

No Windows:

```bash
.venv\Scripts\activate
```

No macOS/Linux:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Executar a aplicação

Na pasta onde estão os ficheiros, execute:

```bash
streamlit run streamlit_app.py
```

A aplicação deverá abrir automaticamente no browser.

## Como funciona

Quando a aplicação é executada pela primeira vez, cria o ficheiro:

```text
adultos_portugal_streamlit.csv
```

Esse ficheiro contém as amostras sintéticas iniciais. Sempre que um estudante submete uma nova observação, a linha é adicionada ao mesmo ficheiro CSV com a coluna:

```text
origem = "Aluno"
```

As amostras criadas automaticamente têm:

```text
origem = "Sintética"
```

Isto permite distinguir visualmente os dados sintéticos dos dados reais introduzidos pelos estudantes.

## Variáveis do dataset

O dataset contém as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `id` | Identificador da observação |
| `idade` | Idade em anos |
| `sexo` | Feminino ou Masculino |
| `altura_cm` | Altura em centímetros |
| `peso_kg` | Peso em quilogramas |
| `imc` | Índice de Massa Corporal |
| `categoria_imc` | Categoria de IMC |
| `origem` | Sintética ou Aluno |

## Nota importante

O dataset sintético serve apenas para fins pedagógicos, demonstração e teste de código. Não deve ser usado para inferência científica sobre adultos em Portugal.
