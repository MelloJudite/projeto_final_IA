"""Gráficos e discussão calculados a partir dos resultados medidos."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taxi_rl import ALGORITHMS, LABELS, SCENARIOS


COLORS = {"q_learning": "#2563a8", "sarsa": "#c75b12"}


def _rolling_runs(history, algorithm, scenario, window):
    group = history.loc[(history.algoritmo == algorithm) & (history.treino == scenario)]
    rewards = group.pivot(index="episodio", columns="execucao", values="reward")
    return rewards.rolling(window, min_periods=window).mean()


def learning_figure(history, window=100):
    """Uma única figura compara os algoritmos nos dois ambientes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True, layout="constrained")
    for ax, scenario in zip(axes, SCENARIOS):
        for algorithm in ALGORITHMS:
            smoothed = _rolling_runs(history, algorithm, scenario, window)
            mean = smoothed.mean(axis=1)
            std = smoothed.std(axis=1, ddof=1).fillna(0)
            ax.plot(mean.index, mean, label=LABELS[algorithm], color=COLORS[algorithm], lw=1.6)
            ax.fill_between(mean.index, mean - std, mean + std,
                            color=COLORS[algorithm], alpha=0.16)
        goal = "Meta seca: reward médio = 8" if scenario == "deterministico" else "Referência seca = 8 (não é meta da chuva)"
        ax.axhline(8, color="#444444", ls="--", lw=1.2, label=goal)
        ax.set(title=LABELS[scenario], xlabel="Episódio de treinamento")
        ax.set_yscale("symlog", linthresh=20)
        ax.set_yticks([-1000, -500, -200, -100, -50, -20, 0, 8, 20])
        ax.set_yticklabels(["−1000", "−500", "−200", "−100", "−50", "−20", "0", "8", "20"])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Retorno por episódio (escala symlog)")
    fig.suptitle(f"Aprendizado: média móvel de {window} episódios; média ± DP entre execuções")
    return fig


def individual_figure(history, window=100):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True, layout="constrained")
    for row, scenario in enumerate(SCENARIOS):
        for col, algorithm in enumerate(ALGORITHMS):
            ax = axes[row, col]
            smoothed = _rolling_runs(history, algorithm, scenario, window)
            for run in smoothed:
                ax.plot(smoothed.index, smoothed[run], lw=0.9, label=f"Execução {run}", alpha=0.85)
            ax.axhline(8, color="#444444", ls="--", lw=1)
            ax.set_title(f"{LABELS[algorithm]} · {LABELS[scenario]}")
            ax.set_yscale("symlog", linthresh=20)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, ncol=2)
            if row == 1:
                ax.set_xlabel("Episódio")
            if col == 0:
                ax.set_ylabel("Retorno (média móvel; symlog)")
    fig.suptitle("Cinco treinamentos independentes por algoritmo e ambiente; referência seca = 8")
    return fig


def validation_figure(checkpoints):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for ax, scenario in zip(axes, SCENARIOS):
        for algorithm in ALGORITHMS:
            group = checkpoints.loc[(checkpoints.algoritmo == algorithm) & (checkpoints.treino == scenario)]
            stats = group.groupby("episodio").reward_medio.agg(["mean", "std"]).fillna(0)
            ax.plot(stats.index, stats["mean"], marker=".", color=COLORS[algorithm], label=LABELS[algorithm])
            ax.fill_between(stats.index, stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"], color=COLORS[algorithm], alpha=0.16)
        ax.axhline(8, color="#444444", ls="--", label="Meta seca = 8" if scenario == "deterministico" else "Referência seca = 8")
        ax.set(title=LABELS[scenario], xlabel="Episódio de treinamento", ylabel="Reward médio sem exploração")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Checkpoints: política gulosa em 100 partidas de validação (média ± DP entre execuções)")
    return fig


def performance_figure(evaluations):
    scenarios = [("deterministico", "deterministico", "Seco → seco"),
                 ("deterministico", "chuva", "Seco → chuva"),
                 ("chuva", "chuva", "Chuva → chuva"),
                 ("chuva", "deterministico", "Chuva → seco")]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    positions = np.arange(len(scenarios))
    for offset, algorithm in zip((-0.18, 0.18), ALGORITHMS):
        means, deviations = [], []
        for training, testing, _ in scenarios:
            group = evaluations.loc[(evaluations.algoritmo == algorithm)
                                    & (evaluations.treino == training) & (evaluations.teste == testing)]
            means.append(group.timesteps.mean())
            deviations.append(group.timesteps.std(ddof=1))
        ax.bar(positions + offset, means, 0.34, yerr=deviations, capsize=4,
               label=LABELS[algorithm], color=COLORS[algorithm], alpha=0.9)
    ax.set_xticks(positions, [item[2] for item in scenarios])
    ax.set(ylabel="Timesteps: média ± DP das partidas", xlabel="Ambiente de treinamento → ambiente de teste",
           title="Avaliação final: 100 partidas por tabela, cinco tabelas por condição")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    return fig


def save_figures(results, output_dir="resultados", window=100):
    destination = Path(output_dir) / "figuras"
    destination.mkdir(parents=True, exist_ok=True)
    builders = {"comparacao_aprendizado": lambda: learning_figure(results["treinamento"], window),
                "cinco_execucoes": lambda: individual_figure(results["treinamento"], window),
                "checkpoints_gulosos": lambda: validation_figure(results["checkpoints"]),
                "desempenho_final": lambda: performance_figure(results["avaliacoes"])}
    paths = {}
    for name, builder in builders.items():
        fig = builder()
        path = destination / f"{name}.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths[name] = path
    return paths


def discussion(results):
    """Texto gerado com valores reais; não presume qual método vencerá."""
    summary = results["resumo_avaliacoes"]
    efficiency = results["eficiencia_aprendizado"]
    by_run = results["resumo_por_execucao"]
    lines = []
    def row(algorithm, training, testing):
        return summary.loc[(summary.algoritmo == algorithm) & (summary.treino == training)
                           & (summary.teste == testing)].iloc[0]

    lines.append("### Itens 1–3: aprendizado e desempenho no ambiente determinístico\n")
    for algorithm in ALGORITHMS:
        record = row(algorithm, "deterministico", "deterministico")
        metrics = efficiency.loc[(efficiency.algoritmo == algorithm) & (efficiency.treino == "deterministico")]
        hits = metrics.primeiro_checkpoint_reward_8.dropna()
        lines.append(
            f"**{LABELS[algorithm]}:** {record.timesteps_medios_todas:.2f} ± "
            f"{record.timesteps_dp_todas:.2f} timesteps, reward médio {record.reward_medio:.2f} "
            f"e sucesso em {record.taxa_sucesso:.1%} das partidas. O retorno médio de treinamento "
            f"foi {metrics.reward_medio_treino.mean():.2f}, e nos últimos 1.000 episódios foi "
            f"{metrics.reward_ultimos_1000.mean():.2f}. A meta de validação 8 foi atingida por "
            f"{len(hits)}/{len(metrics)} execuções" +
            (f", com primeiro checkpoint em média no episódio {hits.mean():.0f}." if len(hits) else ".")
        )
    dry_q = row("q_learning", "deterministico", "deterministico")
    dry_s = row("sarsa", "deterministico", "deterministico")
    dry_metrics = efficiency.loc[efficiency.treino == "deterministico"]
    q_cost = dry_metrics.loc[dry_metrics.algoritmo == "q_learning", "reward_medio_treino"].mean()
    s_cost = dry_metrics.loc[dry_metrics.algoritmo == "sarsa", "reward_medio_treino"].mean()
    lines.append(
        f"Quanto ao custo de aprender no seco, Q-learning − Sarsa = {q_cost - s_cost:+.2f} "
        "unidades de reward médio ao longo do treino (valor positivo favorece Q-learning). "
        "Os checkpoints e os retornos iniciais ajudam a identificar diferenças de velocidade "
        "que podem desaparecer no teste final. Um orçamento comum permite acompanhar também "
        "a estabilização do cenário chuvoso, mesmo que um método resolva o seco antes."
    )
    lines.append(
        f"A diferença observada Sarsa − Q-learning no teste seco é "
        f"{dry_s.timesteps_medios_todas - dry_q.timesteps_medios_todas:+.2f} timesteps. "
        "Ela descreve esta amostra; não demonstra superioridade estatística. O retorno médio "
        "durante todo o treinamento mede o custo da exploração; os checkpoints medem a qualidade "
        "da política gulosa. O primeiro cruzamento da meta pode ser transitório e tem resolução "
        "igual ao intervalo dos checkpoints, portanto não é uma prova de convergência."
    )
    lines.append(
        "A meta é verificada nas sementes de validação. Uma média de teste ligeiramente "
        "abaixo de 8 pode coexistir com checkpoints acima de 8: o teste tem outros estados "
        "iniciais e apenas 100 partidas por tabela, além de possíveis diferenças residuais "
        "da política. A meta não é uma garantia para qualquer amostra de estados."
    )
    lines.append("### Item 4: transferência da política seca para a chuva\n")
    for algorithm in ALGORITHMS:
        dry = row(algorithm, "deterministico", "deterministico")
        transfer = row(algorithm, "deterministico", "chuva")
        lines.append(
            f"**{LABELS[algorithm]}:** sem retreinar, a tabela seca passou de "
            f"{dry.timesteps_medios_todas:.2f} ± {dry.timesteps_dp_todas:.2f} para "
            f"{transfer.timesteps_medios_todas:.2f} ± {transfer.timesteps_dp_todas:.2f} "
            f"timesteps na chuva (variação de {transfer.timesteps_medios_todas - dry.timesteps_medios_todas:+.2f}); "
            f"sucesso {transfer.taxa_sucesso:.1%}, reward médio {transfer.reward_medio:.2f}."
        )
    lines.append(
        "A política consulta o estado novamente depois de cada passo, podendo corrigir desvios. "
        "Isso permite alguma transferência sem novo treinamento, mas caminhos curtos no seco "
        "podem ter maior custo esperado na chuva. Aqui p=0,6 significa 60% de movimento na direção "
        "solicitada e 20% para cada lateral. Paredes podem impedir movimentos sorteados, produzindo "
        "passos sem deslocamento. Pegar e deixar o passageiro continuam determinísticos."
    )
    lines.append("### Item 5: treinamento no ambiente não determinístico\n")
    for algorithm in ALGORITHMS:
        transfer = row(algorithm, "deterministico", "chuva")
        rainy = row(algorithm, "chuva", "chuva")
        dry = row(algorithm, "deterministico", "deterministico")
        improvement = transfer.timesteps_medios_todas - rainy.timesteps_medios_todas
        verb = "reduziu" if improvement >= 0 else "aumentou"
        metrics = efficiency.loc[(efficiency.algoritmo == algorithm) & (efficiency.treino == "chuva")]
        lines.append(
            f"**{LABELS[algorithm]} treinado na chuva:** "
            f"{rainy.timesteps_medios_todas:.2f} ± {rainy.timesteps_dp_todas:.2f} timesteps, "
            f"sucesso {rainy.taxa_sucesso:.1%} e reward médio {rainy.reward_medio:.2f}. "
            f"Retreinar {verb} o custo médio em {abs(improvement):.2f} timesteps em relação "
            f"à transferência da tabela seca. Em relação ao treino/teste secos, a diferença é "
            f"{rainy.timesteps_medios_todas - dry.timesteps_medios_todas:+.2f} timesteps. "
            f"O retorno médio ao longo do treino na chuva foi {metrics.reward_medio_treino.mean():.2f} "
            f"e nos últimos 1.000 episódios, {metrics.reward_ultimos_1000.mean():.2f}."
        )
    rainy_q = row("q_learning", "chuva", "chuva")
    rainy_s = row("sarsa", "chuva", "chuva")
    rainy_metrics = efficiency.loc[efficiency.treino == "chuva"]
    rainy_q_cost = rainy_metrics.loc[rainy_metrics.algoritmo == "q_learning", "reward_medio_treino"].mean()
    rainy_s_cost = rainy_metrics.loc[rainy_metrics.algoritmo == "sarsa", "reward_medio_treino"].mean()
    lines.append(
        f"Na avaliação após treino na chuva, Sarsa − Q-learning = "
        f"{rainy_s.timesteps_medios_todas - rainy_q.timesteps_medios_todas:+.2f} timesteps. "
        f"Durante o aprendizado na chuva, Q-learning − Sarsa = "
        f"{rainy_q_cost - rainy_s_cost:+.2f} unidades de reward médio. "
        "Q-learning é off-policy: seu alvo usa a melhor próxima ação, mesmo que o comportamento "
        "ainda explore. Sarsa é on-policy: usa a próxima ação realmente escolhida pela política "
        "epsilon-greedy e incorpora seu custo de exploração. Conforme epsilon diminui, os alvos "
        "ficam mais próximos. Sarsa não é necessariamente superior na chuva: ambos estimam "
        "retornos esperados a partir de transições estocásticas; Q-learning também considera "
        "essa aleatoriedade, apesar de maximizar sobre ações."
    )
    lines.append("### Diagnóstico adicional: política da chuva aplicada no seco\n")
    for algorithm in ALGORITHMS:
        reverse = row(algorithm, "chuva", "deterministico")
        lines.append(
            f"**{LABELS[algorithm]} (chuva → seco):** "
            f"{reverse.timesteps_medios_todas:.2f} ± {reverse.timesteps_dp_todas:.2f} "
            f"timesteps observados, sucesso {reverse.taxa_sucesso:.1%} e "
            f"{int(reverse.n_truncamentos)} truncamentos. Entre as partidas concluídas, "
            f"a média foi {reverse.timesteps_medios_sucesso:.2f} timesteps."
        )
    lines.append(
        "Transferência não é simétrica. Uma ação cujo movimento pretendido é bloqueado "
        "pode ainda provocar deslocamento lateral na chuva; no seco, essa mesma ação "
        "pode manter o táxi parado indefinidamente. Isso oferece uma explicação possível "
        "para falhas de transferência reversa, junto de valores ainda imprecisos após "
        "treinamento finito. Identificar a causa de cada falha exigiria inspecionar suas "
        "trajetórias; a taxa de sucesso, por si só, não distingue essas causas."
    )
    lines.append("### Variabilidade e limites\n")
    for algorithm in ALGORITHMS:
        for scenario in SCENARIOS:
            records = by_run.loc[(by_run.algoritmo == algorithm) & (by_run.treino == scenario)
                                 & (by_run.teste == scenario)]
            lines.append(
                f"{LABELS[algorithm]} / {LABELS[scenario]}: o DP entre as médias das "
                f"{len(records)} tabelas é {records.timesteps_medios_todas.std(ddof=1):.3f} timesteps."
            )
    lines.append(
        "O DP das partidas mistura a variação dos estados iniciais, das transições na chuva e "
        "das tabelas. O DP entre médias das tabelas descreve separadamente a sensibilidade ao "
        "treinamento. As 500 partidas agrupadas por condição são cinco blocos de 100, com as "
        "mesmas sementes de teste entre tabelas; não são 500 réplicas independentes do aprendizado. "
        "As sementes iguais tornam os estados iniciais comparáveis, mas políticas diferentes "
        "produzem trajetórias distintas, e não garantem os mesmos desvios em cada estado. "
        "Falhas são censuradas no limite de passos: por isso as tabelas informam taxa de sucesso "
        "e médias de timesteps tanto de todas as partidas quanto somente das concluídas. "
        "Hiperparâmetros fixados antes dos testes; nenhuma tabela foi selecionada pelo resultado "
        "do teste. A execução 1 é a representante predefinida."
    )
    lines.append("### Conclusões\n")
    lines.append(
        "Os experimentos separam três efeitos: aprender com exploração, executar uma política "
        "gulosa e transferir essa política para outra dinâmica. A chuva adiciona desvios e passos "
        "sem progresso, alterando o custo esperado das rotas e a variabilidade dos retornos. "
        "As medições acima mostram se o retreinamento compensou esse efeito neste orçamento de "
        "episódios. Retorno negativo na chuva não implica necessariamente falha: sem ações "
        "ilegais, uma entrega bem-sucedida em T passos tem retorno 21 − T. "
        "A meta 8 do ambiente seco não é uma exigência para a chuva. Os resultados são "
        "evidência empírica de políticas aprendidas; treinamento finito, taxa de aprendizado "
        "decrescente e epsilon mínimo positivo não certificam a política ótima nem satisfazem "
        "automaticamente todas as hipóteses de convergência assintótica."
    )
    return "\n\n".join(lines)
