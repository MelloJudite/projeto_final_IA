"""Cria o notebook e, opcionalmente, executa suas células com nbclient."""

import argparse
from importlib.metadata import distributions
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []
    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip()))
    def code(text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip()))

    md(r"""
    # Projeto 2 — Q-learning e Sarsa no Taxi-v4

    **Conceitos básicos de aprendizado por reforço e ambientes não determinísticos**

    Este relatório resolve os itens 1 a 5 com Gymnasium. Compara cinco treinamentos
    independentes de cada algoritmo no ambiente determinístico e na chuva, acompanha
    o retorno de cada episódio e avalia as tabelas aprendidas em partidas novas.
    Os resultados e a discussão abaixo são calculados a partir dos experimentos executados.

    | Parte do enunciado | Evidência neste notebook |
    |---|---|
    | 1. Cinco treinamentos Q-learning e meta de reward | Histórico completo, curvas individuais e referência 8 |
    | 2–3. Avaliação em 100 partidas e comparação com Sarsa | Resumo da execução 1, cinco execuções e gráfico conjunto |
    | 4. Tabela seca aplicada na chuva | Teste seco → chuva, sem atualizar a tabela |
    | 5. Treinamento na chuva | Cinco novos treinamentos de cada algoritmo e comparação |

    Os módulos `taxi_rl.py` e `taxi_report.py` contêm a implementação reutilizada aqui.
    Este notebook pode carregar os dados entregues ou refazer todos os treinamentos.
    """)
    md(r"""
    ## 1. Ambiente, recompensas e meta

    O Taxi-v4 possui **500 estados codificados** e **6 ações**: sul, norte, leste,
    oeste, pegar e deixar passageiro. Há 300 estados iniciais possíveis. A recompensa
    é −1 por passo comum, +20 pela entrega correta e −10 por tentativa ilegal de
    pegar/deixar. O retorno de um episódio é a soma **não descontada** das recompensas;
    o desconto γ é usado apenas na atualização dos valores Q.

    Uma entrega em T passos, sem ações ilegais, tem retorno **21 − T**. Portanto,
    +20 é a recompensa da entrega, não uma meta de retorno médio por episódio.
    O registro oficial estabelece **reward_threshold = 8** para o Taxi-v4 padrão.
    Usamos retorno médio ≥ 8 como referência operacional do ambiente seco, e não
    como exigência para cada episódio ou certificado de política ótima.

    Na chuva, `is_rainy=True, rainy_probability=0.6` significa 60% de probabilidade
    de tentar a direção solicitada e 20% para cada lateral. Paredes ainda bloqueiam
    movimentos; pegar e deixar continuam determinísticos. Não ativamos passageiro
    volúvel. O limiar 8 continua no registro, mas não foi recalibrado para a chuva;
    aparece nesse cenário apenas como referência do ambiente seco.

    Preservamos o wrapper `TimeLimit` de **200 passos**. Não usamos `.env` para
    remover wrappers nem `render_mode="human"` nos experimentos: renderização não
    participa do aprendizado e tornaria as repetições mais lentas.

    Fontes: [documentação oficial do Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/),
    [registro do Taxi-v4 no Gymnasium 1.3.0](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/__init__.py),
    [tratamento de limites de tempo](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
    """)
    code("""
    from dataclasses import asdict
    from pathlib import Path
    import inspect
    import json
    import sys
    import gymnasium as gym
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image, Markdown, display

    from taxi_rl import (
        Config, LABELS, make_env, train, evaluate, td_update, epsilon_greedy,
        run_experiments, load_results, demonstrate,
    )
    from taxi_report import save_figures, discussion

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    CONFIG = Config()
    OUTPUT = Path("resultados")
    print(f"Python {sys.version.split()[0]} | Gymnasium {gym.__version__}")
    print(f"Ambiente: Taxi-v4 | meta seca: {gym.spec('Taxi-v4').reward_threshold}")
    """)
    md(r"""
    ## 2. Protocolo experimental e hiperparâmetros

    São **20 treinamentos**: 2 algoritmos × 2 ambientes × 5 sementes. Cada tabela
    começa em zero e aprende por 20.000 episódios. O mesmo orçamento, as mesmas
    sementes por execução e as mesmas regras de α e ε valem para todos os grupos.
    As seis ações estão sempre disponíveis: não usamos `action_mask` nem conhecimento
    prévio do mapa para escolher ações.

    - **γ = 0,99:** considera o custo de trajetos longos.
    - **α inicial = 0,6:** aprendizado rápido nas primeiras visitas; para a n-ésima
      visita a (s,a), αₙ = 0,6 / [1 + 0,01(n − 1)]⁰·⁶. A redução por visita diminui
      a influência de transições isoladas na chuva sem alterar o protocolo entre algoritmos.
    - **ε-greedy:** εₑ = max(0,005; 1,0 × 0,9995ᵉ), com e começando em zero.
      Há exploração ampla no início e pouca exploração no final. Empates durante
      o treino são sorteados para não privilegiar a primeira ação.
    - **Validação:** a cada 1.000 episódios, 100 partidas gulosas em ambiente separado;
      não atualizam Q nem mudam o sorteio do treino. Não há parada antecipada ou
      escolha de hiperparâmetros/tabelas a partir do teste.
    - **Teste final:** ε = 0; 100 partidas por tabela e cenário, com desempate por
      menor índice. A **execução 1** é a representante definida antecipadamente;
      também mostramos todas as cinco para medir variabilidade. São 4.000 testes,
      incluindo chuva → seco como diagnóstico adicional.
    - **Sementes separadas:** treino 2026–2030; teste 1.000.000–1.000.099;
      validação 2.000.000–2.000.099. Os métodos recebem os mesmos estados iniciais
      de teste; trajetórias podem divergir conforme suas ações.

    Estes valores são uma escolha prática, não o resultado de uma busca exaustiva.
    O primeiro checkpoint que alcança 8 indica velocidade de aprendizado no seco,
    com resolução de 1.000 episódios; não comprova convergência permanente.
    """)
    code("""
    display(pd.DataFrame(asdict(CONFIG).items(), columns=["Hiperparâmetro", "Valor"]))
    """)
    md(r"""
    ## 3. Algoritmos e implementação

    **Q-learning (off-policy):**

    $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_{t+1}
    + \gamma\max_a Q(s_{t+1},a) - Q(s_t,a_t)\right].$$

    **Sarsa (on-policy):** primeiro sorteia a próxima ação pela política ε-greedy;
    essa é a ação usada no alvo e executada no passo seguinte:

    $$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_{t+1}
    + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\right].$$

    Em uma entrega (`terminated=True`), o alvo é somente a recompensa.
    Um limite externo (`truncated=True`) encerra a coleta, mas mantém o bootstrap
    do próximo estado. Tratar truncamento como estado terminal introduziria viés.
    Abaixo estão as funções efetivamente usadas, carregadas do módulo para que
    a explicação e a implementação permaneçam consistentes.
    """)
    code("""
    for function in (epsilon_greedy, td_update, train):
        display(Markdown(f"### `{function.__name__}`\\n```python\\n{inspect.getsource(function)}\\n```"))
    """)
    md("""
    ## 4. Execução dos experimentos

    Com `REFAZER_TREINAMENTOS = False`, os dados salvos são reutilizados após
    conferir a configuração. Altere para `True` para refazer as vinte tabelas.
    Se os resultados ainda não existirem, o notebook os calcula automaticamente.
    Os arquivos CSV mantêm cada episódio de treinamento e cada partida de teste;
    as tabelas Q são salvas como `.npy`.
    """)
    code("""
    REFAZER_TREINAMENTOS = False
    if REFAZER_TREINAMENTOS or not (OUTPUT / "metadados.json").exists():
        resultados = run_experiments(CONFIG, OUTPUT)
    else:
        metadata = json.loads((OUTPUT / "metadados.json").read_text(encoding="utf-8"))
        if metadata["config"] != asdict(CONFIG):
            raise ValueError("Configuração salva diferente. Ative REFAZER_TREINAMENTOS.")
        if metadata["gymnasium"] != gym.__version__:
            raise ValueError("Use a versão registrada do Gymnasium ou refaça o treinamento.")
        resultados = load_results(OUTPUT)

    historico = resultados["treinamento"]
    avaliacoes = resultados["avaliacoes"]
    checkpoints = resultados["checkpoints"]
    eficiencia = resultados["eficiencia_aprendizado"]
    por_execucao = resultados["resumo_por_execucao"]
    resumo = resultados["resumo_avaliacoes"]
    print(f"{len(historico):,} episódios registrados; {len(avaliacoes):,} partidas de teste.")
    print(f"Tabelas salvas: {len(list((OUTPUT / 'q_tables').glob('*.npy')))}")
    display(historico.head())
    """)
    md("""
    ## 5. Curvas de aprendizado — itens 1, 3 e 5

    A figura principal reúne Q-learning e Sarsa; cada painel corresponde a um
    ambiente. Para cada execução calculamos uma média móvel de 100 episódios e
    então a média entre as cinco execuções. A faixa mostra **±1 desvio padrão
    amostral entre as cinco curvas suavizadas**, não intervalo de confiança.
    Os primeiros 99 episódios não aparecem na suavização, mas estão no CSV.

    O eixo de reward usa **symlog**, linear entre −20 e +20 e logarítmico fora
    dessa faixa, para mostrar tanto as penalidades iniciais quanto a aproximação
    da meta. Essa transformação afeta apenas o desenho, não os dados ou as médias.
    O retorno mostrado é o retorno por episódio, não sua soma ao longo do treinamento.
    """)
    code("""
    figuras = save_figures(resultados, OUTPUT, CONFIG.smoothing_window)
    display(Image(filename=str(figuras["comparacao_aprendizado"])))
    """)
    md("""
    As cinco curvas individuais permitem identificar execuções instáveis que
    poderiam ficar escondidas pela média. A linha tracejada é a referência seca 8.
    """)
    code("""
    display(Image(filename=str(figuras["cinco_execucoes"])))
    """)
    md("""
    O reward de treinamento inclui exploração. Os checkpoints abaixo usam política
    gulosa em sementes de **validação**, permitindo verificar a meta sem confundir
    exploração com desempenho aprendido. O teste final permanece separado.
    """)
    code("""
    display(Image(filename=str(figuras["checkpoints_gulosos"])))
    display(eficiencia)
    """)
    md("""
    ## 6. Avaliação final — 100 partidas por tabela

    Cada passo, inclusive pegar e deixar, conta como um timestep. O desvio padrão
    é **amostral (ddof=1)**. A tabela seguinte responde diretamente aos itens 2 e 4
    usando a execução 1, escolhida antes de observar os resultados.

    As colunas `*_todas` incluem partidas truncadas com o número de passos observados;
    são censuradas no limite de 200, não entregas concluídas. As colunas `*_sucesso`
    consideram somente entregas efetivamente concluídas. Leia as médias junto da
    taxa de sucesso: uma política que falha não pode parecer eficiente só por
    encerrar a coleta no limite. O CSV `avaliacoes.csv` registra cada um dos 100 testes.
    """)
    code("""
    colunas = ["algoritmo", "treino", "teste", "n_partidas", "taxa_sucesso",
               "n_truncamentos", "timesteps_medios_todas", "timesteps_dp_todas",
               "timesteps_medios_sucesso", "timesteps_dp_sucesso", "reward_medio"]
    representante = por_execucao.loc[por_execucao.execucao == 1, colunas]
    display(representante)
    display(avaliacoes.loc[(avaliacoes.execucao == 1)
                          & (avaliacoes.algoritmo == "q_learning")
                          & (avaliacoes.treino == "deterministico")
                          & (avaliacoes.teste == "deterministico")].head(10))
    """)
    md("""
    Para verificar a consistência entre sementes, apresentamos também os cinco
    blocos de 100 partidas separadamente e o resumo agrupado (500 partidas por
    condição). Os cinco blocos compartilham as sementes de teste; não representam
    500 treinamentos independentes. O gráfico usa média ± DP dos timesteps das partidas.
    """)
    code("""
    display(por_execucao[["execucao"] + colunas])
    display(resumo[colunas])
    display(Image(filename=str(figuras["desempenho_final"])))
    """)
    md("""
    ## 7. Impacto da chuva e do retreinamento

    **Transferência:** compara seco → chuva com seco → seco, mantendo a mesma tabela.
    **Retreinamento:** compara chuva → chuva com seco → chuva. Valores negativos nas
    diferenças de timesteps indicam menor custo na segunda condição comparada.
    As comparações são calculadas por tabela/execução para preservar os cinco blocos.
    """)
    code("""
    comparacoes = []
    for (algoritmo, execucao), grupo in por_execucao.groupby(["algoritmo", "execucao"]):
        valores = grupo.set_index(["treino", "teste"])
        seco = valores.loc[("deterministico", "deterministico")]
        transferencia = valores.loc[("deterministico", "chuva")]
        retreinado = valores.loc[("chuva", "chuva")]
        comparacoes.append({
            "algoritmo": algoritmo, "execucao": execucao,
            "delta_chuva_sem_retreino": transferencia.timesteps_medios_todas - seco.timesteps_medios_todas,
            "delta_retreinar_na_chuva": retreinado.timesteps_medios_todas - transferencia.timesteps_medios_todas,
            "delta_chuva_treinada_vs_seco": retreinado.timesteps_medios_todas - seco.timesteps_medios_todas,
        })
    comparacoes = pd.DataFrame(comparacoes)
    comparacoes.to_csv(OUTPUT / "comparacoes.csv", index=False)
    display(comparacoes)
    display(comparacoes.groupby("algoritmo")[list(comparacoes.columns[2:])].agg(["mean", "std"]))
    """)
    md("""
    ## 8. Discussão dos resultados e conclusões

    O texto seguinte é gerado a partir dos valores medidos, incluindo diferenças,
    dispersões e alcance da meta. Assim, executar novamente com outra configuração
    não mantém conclusões numéricas desatualizadas.
    """)
    code("""
    display(Markdown(discussion(resultados)))
    """)
    md("""
    ## 9. Reutilização da Q-table e arquivos de entrega

    A função abaixo carrega uma tabela e executa uma partida com renderização textual.
    Para visualizar uma janela, instale `pygame-ce` e use `render_mode="human"`.
    Essa visualização é opcional e não interfere nos experimentos.
    """)
    code("""
    tabela = OUTPUT / "q_tables" / "q_learning_deterministico_execucao_1.npy"
    print("Exemplo de uso (opcional):")
    print(f"demonstrate({str(tabela)!r}, scenario='chuva', render_mode='ansi')")
    # Descomente para acompanhar uma partida:
    # demonstrate(tabela, scenario="chuva", render_mode="ansi")
    display(pd.DataFrame([
        {"Arquivo": path.name, "Tamanho (KB)": round(path.stat().st_size / 1024, 1)}
        for path in sorted(OUTPUT.glob("*.csv"))
    ]))
    print((OUTPUT / "metadados.json").read_text(encoding="utf-8"))
    """)
    md("""
    ## 10. Verificação da entrega

    As verificações abaixo conferem a contagem de episódios, os 100 testes de cada
    tabela/ambiente, o limite de passos e a relação entre retorno e entrega.
    Os testes automatizados em `tests/test_taxi_rl.py` verificam, adicionalmente,
    alvos TD, truncamento, ação seguinte do Sarsa, chuva e reprodutibilidade.
    """)
    code("""
    assert len(historico) == 2 * 2 * CONFIG.runs * CONFIG.episodes
    assert len(avaliacoes) == 2 * 2 * 2 * CONFIG.runs * CONFIG.evaluation_episodes
    assert (historico.groupby(["algoritmo", "treino", "execucao"]).size() == CONFIG.episodes).all()
    assert (avaliacoes.groupby(["algoritmo", "treino", "teste", "execucao"]).size()
            == CONFIG.evaluation_episodes).all()
    assert avaliacoes.timesteps.between(1, CONFIG.max_steps).all()
    assert (avaliacoes.sucesso | avaliacoes.truncado).all()
    concluidas = avaliacoes.loc[avaliacoes.sucesso]
    np.testing.assert_allclose(concluidas.reward,
                               21 - concluidas.timesteps - 9 * concluidas.acoes_ilegais)
    assert all(np.isfinite(np.load(path, allow_pickle=False)).all()
               for path in (OUTPUT / "q_tables").glob("*.npy"))
    print("Verificação concluída: dados completos, tabelas finitas e métricas consistentes.")
    """)
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "file_extension": ".py"}
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build_notebook()
    destination = ROOT / "relatorio_taxi.ipynb"
    nbformat.write(notebook, destination)
    if args.execute:
        client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(notebook, destination)
        versions = sorted({f"{package.metadata['Name']}=={package.version}"
                           for package in distributions()})
        (ROOT / "requirements-lock.txt").write_text(
            "# Versões do ambiente utilizado na entrega (Windows / Python 3.13).\n"
            + "\n".join(versions) + "\n", encoding="utf-8")
    nbformat.validate(notebook)
    print(f"Notebook salvo: {destination.name}")


if __name__ == "__main__":
    main()
