# Projeto 2 — Q-learning e Sarsa no Taxi-v4

Resolução com Gymnasium dos experimentos determinísticos e com chuva
(`is_rainy=True, rainy_probability=0.6`). O relatório principal é
[relatorio_taxi.ipynb](relatorio_taxi.ipynb), entregue com células executadas,
gráficos, tabelas e discussão dos resultados medidos.

São cinco treinamentos independentes por algoritmo e ambiente: 20 tabelas,
20.000 episódios por tabela. Cada tabela é avaliada em 100 partidas novas em
cada ambiente, totalizando 4.000 avaliações. A execução 1 é a representante
predefinida para as tabelas de 100 partidas; as cinco execuções também são
apresentadas individualmente e agrupadas.

## Instalação e abertura

Requer Python 3.11 ou superior. No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab relatorio_taxi.ipynb
```

Selecione o Python do ambiente `.venv` ao abrir no VS Code. O notebook pode ser
lido com os resultados já salvos. Ao executar suas células, reutiliza os dados
compatíveis; altere `REFAZER_TREINAMENTOS = True` para refazer tudo. Para reproduzir
as versões exatas utilizadas na entrega, use `requirements-lock.txt` no lugar
de `requirements.txt` (o lock foi gerado em Windows / Python 3.13).

## Executar pelo terminal

```powershell
# Repete os experimentos completos e grava os dados e as Q-tables.
.\.venv\Scripts\python.exe -u taxi_rl.py

# Recria e executa o notebook, gerando também os gráficos.
.\.venv\Scripts\python.exe gerar_notebook.py --execute

# Testa os algoritmos e o protocolo experimental.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Acompanha uma partida em texto usando uma tabela aprendida no seco.
.\.venv\Scripts\python.exe taxi_rl.py --demo resultados/q_tables/q_learning_deterministico_execucao_1.npy --scenario chuva
```

Para uma execução curta de verificação, preservando a entrega completa:

```powershell
.\.venv\Scripts\python.exe taxi_rl.py --episodes 50 --runs 1 --output resultados_teste
```

A janela gráfica é opcional: instale `pygame-ce` e chame
`demonstrate(caminho_tabela, scenario="chuva", render_mode="human")`.
Os experimentos usam o ambiente sem renderização para reduzir o tempo de execução.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `relatorio_taxi.ipynb` | Relatório completo com resultados e conclusões |
| `taxi_rl.py` | Q-learning, Sarsa, treinamento, avaliação e demonstração |
| `taxi_report.py` | Gráficos e discussão calculada a partir dos dados |
| `gerar_notebook.py` | Geração e execução reproduzível do notebook |
| `tests/test_taxi_rl.py` | Testes de alvos TD, truncamento, Sarsa, chuva e sementes |
| `resultados/treinamento.csv` | Reward, timesteps e epsilon de cada episódio/execução |
| `resultados/avaliacoes.csv` | Cada partida de teste: passos, reward, sucesso e truncamento |
| `resultados/checkpoints.csv` | Avaliação gulosa em sementes de validação durante o treino |
| `resultados/eficiencia_aprendizado.csv` | Custo de aprender, checkpoints da meta e duração |
| `resultados/resumo_por_execucao.csv` | Média e DP das 100 partidas de cada tabela/cenário |
| `resultados/resumo_avaliacoes.csv` | Resumos agrupados das cinco tabelas |
| `resultados/comparacoes.csv` | Diferenças causadas pela chuva e pelo retreinamento |
| `resultados/metadados.json` | Configuração, versões e regras de avaliação |
| `resultados/q_tables/` | Vinte tabelas Q em formato NumPy `.npy` |
| `resultados/figuras/` | Gráficos PNG reutilizáveis |

## Decisões do experimento

Os algoritmos usam os mesmos hiperparâmetros, as seis ações sem mascaramento,
taxa de aprendizado decrescente por visita e exploração epsilon-greedy decrescente.
O teste é guloso, sem aprendizado e com sementes separadas das do treino e da
validação. Não há seleção de tabela a partir dos resultados de teste.

O ambiente mantém o limite de 200 passos. Truncamento encerra a coleta, mas
preserva bootstrap na atualização; uma entrega real remove o bootstrap. Todos
os desvios padrão são amostrais (`ddof=1`). As tabelas distinguem timesteps de
todas as partidas e de partidas concluídas e mostram a taxa de sucesso.

A meta de retorno médio **8** vem do registro oficial do Taxi-v4 padrão; ela não
é a recompensa da entrega (+20) nem uma meta recalibrada para chuva. Na chuva,
`rainy_probability=0.6` significa 60% de movimento na direção solicitada e 20%
para cada lateral. O notebook discute os limites de transferência e o efeito
da aleatoriedade sem presumir qual algoritmo terá melhor desempenho.

Referências: [Taxi — Gymnasium](https://gymnasium.farama.org/environments/toy_text/taxi/),
[registro do ambiente](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/__init__.py),
[limites de tempo e bootstrap](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
