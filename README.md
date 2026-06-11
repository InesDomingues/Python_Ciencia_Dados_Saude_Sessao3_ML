# Classificação Supervisionada

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
