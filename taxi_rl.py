"""Q-learning e Sarsa tabulares para o Taxi-v4, sem mascaramento de ações.

As duas implementações compartilham a política epsilon-greedy e os mesmos
hiperparâmetros. Somente a construção do alvo TD muda entre os algoritmos.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd


ALGORITHMS = ("q_learning", "sarsa")
SCENARIOS = ("deterministico", "chuva")
LABELS = {"q_learning": "Q-learning", "sarsa": "Sarsa",
          "deterministico": "Determinístico", "chuva": "Chuva (p=0,6)"}


@dataclass(frozen=True)
class Config:
    episodes: int = 20_000
    runs: int = 5
    evaluation_episodes: int = 100
    alpha: float = 0.6
    alpha_decay: float = 0.01
    alpha_power: float = 0.6
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.005
    epsilon_decay: float = 0.9995
    max_steps: int = 200
    rainy_probability: float = 0.6
    seed: int = 2026
    evaluation_seed: int = 1_000_000
    validation_seed: int = 2_000_000
    checkpoint_every: int = 1_000
    validation_episodes: int = 100
    smoothing_window: int = 100

    def __post_init__(self):
        for field in ("episodes", "runs", "evaluation_episodes", "max_steps",
                      "checkpoint_every", "validation_episodes", "smoothing_window"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} deve ser positivo.")
        if not 0 < self.alpha <= 1 or self.alpha_decay < 0 or self.alpha_power < 0:
            raise ValueError("Taxa de aprendizado inválida.")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma deve estar entre 0 e 1.")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("Limites de epsilon inválidos.")
        if not 0 < self.epsilon_decay <= 1 or not 0 <= self.rainy_probability <= 1:
            raise ValueError("Probabilidades inválidas.")
        if self.seed < 0 or self.evaluation_seed < 0 or self.validation_seed < 0:
            raise ValueError("As sementes devem ser não negativas.")
        train_seeds = set(range(self.seed, self.seed + self.runs))
        test_seeds = set(range(self.evaluation_seed,
                               self.evaluation_seed + self.evaluation_episodes))
        validation_seeds = set(range(self.validation_seed,
                                     self.validation_seed + self.validation_episodes))
        if train_seeds & test_seeds or train_seeds & validation_seeds or test_seeds & validation_seeds:
            raise ValueError("Sementes de treino, validação e teste devem ser separadas.")


def make_env(scenario: str, config: Config, render_mode=None):
    """Preserva o TimeLimit; não remove wrappers com .env ou .unwrapped."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Cenário desconhecido: {scenario}")
    try:
        return gym.make(
            "Taxi-v4", is_rainy=scenario == "chuva",
            rainy_probability=config.rainy_probability,
            fickle_passenger=False, render_mode=render_mode,
            max_episode_steps=config.max_steps,
        )
    except gym.error.VersionNotFound as error:
        raise RuntimeError("Taxi-v4 requer Gymnasium >= 1.3.0. Instale requirements.txt.") from error


def epsilon_at(episode: int, config: Config) -> float:
    """episode começa em zero."""
    return max(config.epsilon_min,
               config.epsilon_start * config.epsilon_decay**episode)


def epsilon_greedy(q_row: np.ndarray, epsilon: float,
                   rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    # Empates são sorteados no treinamento, evitando favorecer a ação 0.
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[rng.integers(len(best))])


def td_update(q_table, state, action, reward, next_state, terminated,
              algorithm, alpha, gamma, next_action=None):
    """Não anula o bootstrap em truncamento externo; só em terminação real."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")
    target = float(reward)
    if not terminated:
        if algorithm == "q_learning":
            target += gamma * float(q_table[next_state].max())
        else:
            if next_action is None:
                raise ValueError("Sarsa precisa da próxima ação da política.")
            target += gamma * float(q_table[next_state, next_action])
    q_table[state, action] += alpha * (target - q_table[state, action])
    return target


def evaluate(q_table: np.ndarray, scenario: str, config: Config,
             seeds=None) -> pd.DataFrame:
    """Política gulosa, sem atualização da tabela; uma linha por partida.

    No teste os empates usam np.argmax (menor índice), garantindo uma política
    fixa. Timesteps de partidas truncadas são observações censuradas em 200.
    """
    if seeds is None:
        seeds = range(config.evaluation_seed,
                      config.evaluation_seed + config.evaluation_episodes)
    env = make_env(scenario, config)
    rows = []
    try:
        if q_table.shape != (env.observation_space.n, env.action_space.n):
            raise ValueError("Dimensão da Q-table incompatível com o ambiente.")
        for episode, seed in enumerate(seeds, start=1):
            state, _ = env.reset(seed=int(seed))
            total_reward = 0.0
            illegal_actions = 0
            terminated = truncated = False
            for steps in range(1, config.max_steps + 1):
                action = int(q_table[state].argmax())
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                illegal_actions += int(reward == -10)
                if terminated or truncated:
                    break
            rows.append({"episodio": episode, "seed_avaliacao": int(seed),
                         "timesteps": steps, "reward": total_reward,
                         "sucesso": bool(terminated), "truncado": bool(truncated),
                         "acoes_ilegais": illegal_actions})
    finally:
        env.close()
    return pd.DataFrame(rows)


def train(algorithm: str, scenario: str, config: Config, seed: int):
    """Treina uma tabela nova e retorna tabela, histórico e checkpoints."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo desconhecido: {algorithm}")
    env = make_env(scenario, config)
    rng = np.random.default_rng(seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    visits = np.zeros_like(q_table, dtype=np.int64)
    rows, checkpoints = [], []
    try:
        for episode in range(config.episodes):
            # O gerador do ambiente é independente do gerador da política.
            state, _ = env.reset(seed=seed if episode == 0 else None)
            epsilon = epsilon_at(episode, config)
            action = epsilon_greedy(q_table[state], epsilon, rng)
            total_reward = 0.0
            terminated = truncated = False
            for steps in range(1, config.max_steps + 1):
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Sarsa escolhe uma vez e executa exatamente essa próxima ação.
                next_action = None
                if algorithm == "sarsa" and not terminated:
                    next_action = epsilon_greedy(q_table[next_state], epsilon, rng)
                visits[state, action] += 1
                alpha = config.alpha / (
                    1 + config.alpha_decay * (visits[state, action] - 1)
                ) ** config.alpha_power
                td_update(q_table, state, action, reward, next_state, terminated,
                          algorithm, alpha, config.gamma, next_action)
                total_reward += reward
                if terminated or truncated:
                    break
                state = next_state
                action = (next_action if algorithm == "sarsa" else
                          epsilon_greedy(q_table[state], epsilon, rng))
            rows.append({"episodio": episode + 1, "reward": total_reward,
                         "timesteps": steps, "epsilon": epsilon,
                         "sucesso": bool(terminated), "truncado": bool(truncated)})
            if (episode + 1) % config.checkpoint_every == 0 or episode + 1 == config.episodes:
                validation = evaluate(
                    q_table, scenario, config,
                    range(config.validation_seed,
                          config.validation_seed + config.validation_episodes),
                )
                checkpoints.append({
                    "episodio": episode + 1,
                    "reward_medio": validation.reward.mean(),
                    "timesteps_medios": validation.timesteps.mean(),
                    "taxa_sucesso": validation.sucesso.mean(),
                })
    finally:
        env.close()
    return q_table, pd.DataFrame(rows), pd.DataFrame(checkpoints)


def summarize_evaluations(evaluations: pd.DataFrame, by_run=False) -> pd.DataFrame:
    keys = ["algoritmo", "treino", "teste"] + (["execucao"] if by_run else [])
    rows = []
    for key, group in evaluations.groupby(keys, sort=False):
        successful = group.loc[group.sucesso, "timesteps"]
        row = dict(zip(keys, key))
        row.update({
            "n_partidas": len(group), "taxa_sucesso": group.sucesso.mean(),
            "n_truncamentos": int(group.truncado.sum()),
            "timesteps_medios_todas": group.timesteps.mean(),
            "timesteps_dp_todas": group.timesteps.std(ddof=1),
            "timesteps_medios_sucesso": successful.mean(),
            "timesteps_dp_sucesso": successful.std(ddof=1),
            "reward_medio": group.reward.mean(), "reward_dp": group.reward.std(ddof=1),
            "acoes_ilegais_medias": group.acoes_ilegais.mean(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def learning_metrics(history, checkpoints, timings, config):
    rows = []
    for key, group in history.groupby(["algoritmo", "treino", "execucao"], sort=False):
        algorithm, scenario, run = key
        validation = checkpoints.loc[
            (checkpoints.algoritmo == algorithm) & (checkpoints.treino == scenario)
            & (checkpoints.execucao == run)]
        hits = validation.loc[validation.reward_medio >= 8, "episodio"]
        # A média de todos os retornos mede o custo de aprender (sem suavização).
        rows.append({"algoritmo": algorithm, "treino": scenario, "execucao": run,
                     "reward_medio_treino": group.reward.mean(),
                     "reward_primeiros_1000": group.head(1000).reward.mean(),
                     "reward_ultimos_1000": group.tail(1000).reward.mean(),
                     "timesteps_treino": int(group.timesteps.sum()),
                     "primeiro_checkpoint_reward_8": (
                         int(hits.iloc[0]) if scenario == "deterministico" and len(hits) else np.nan),
                     "segundos_treino_validacao": timings[key]})
    return pd.DataFrame(rows)


def run_experiments(config=Config(), output_dir="resultados"):
    """20 treinamentos; 100 testes por tabela e por cenário (4.000 partidas)."""
    output = Path(output_dir)
    tables_dir = output / "q_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    histories, evaluations, checkpoints, timings = [], [], [], {}
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            for run in range(1, config.runs + 1):
                seed = config.seed + run - 1
                start = perf_counter()
                q_table, history, validation = train(algorithm, scenario, config, seed)
                elapsed = perf_counter() - start
                timings[algorithm, scenario, run] = elapsed
                for frame in (history, validation):
                    frame["algoritmo"] = algorithm
                    frame["treino"] = scenario
                    frame["execucao"] = run
                    frame["seed_treino"] = seed
                histories.append(history)
                checkpoints.append(validation)
                np.save(tables_dir / f"{algorithm}_{scenario}_execucao_{run}.npy", q_table)
                for test_scenario in SCENARIOS:
                    evaluation = evaluate(q_table, test_scenario, config)
                    evaluation["algoritmo"] = algorithm
                    evaluation["treino"] = scenario
                    evaluation["teste"] = test_scenario
                    evaluation["execucao"] = run
                    evaluation["seed_treino"] = seed
                    evaluations.append(evaluation)
                print(f"{LABELS[scenario]} | {LABELS[algorithm]} | "
                      f"execução {run}/{config.runs}: {elapsed:.1f}s", flush=True)
    history = pd.concat(histories, ignore_index=True)
    evaluation = pd.concat(evaluations, ignore_index=True)
    validation = pd.concat(checkpoints, ignore_index=True)
    metrics = learning_metrics(history, validation, timings, config)
    frames = {"treinamento": history, "avaliacoes": evaluation,
              "checkpoints": validation, "eficiencia_aprendizado": metrics,
              "resumo_por_execucao": summarize_evaluations(evaluation, by_run=True),
              "resumo_avaliacoes": summarize_evaluations(evaluation)}
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    metadata = {"config": asdict(config), "gymnasium": gym.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "python": platform.python_version(), "platform": platform.platform(),
                "desvio_padrao": "amostral (ddof=1)",
                "politica_teste": "gulosa, epsilon=0, empates por menor índice",
                "representante": "execução 1, definida antes do treinamento",
                "mascara_acoes": False, "reward_threshold_deterministico": gym.spec("Taxi-v4").reward_threshold}
    (output / "metadados.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames


def load_results(output_dir="resultados"):
    output = Path(output_dir)
    names = ("treinamento", "avaliacoes", "checkpoints", "eficiencia_aprendizado",
             "resumo_por_execucao", "resumo_avaliacoes")
    return {name: pd.read_csv(output / f"{name}.csv") for name in names}


def demonstrate(table_path, scenario="deterministico", seed=42, render_mode="ansi",
                config=Config()):
    """Uma partida opcional usando a tabela salva. human requer pygame-ce."""
    q_table = np.load(table_path, allow_pickle=False)
    env = make_env(scenario, config, render_mode)
    total_reward = 0
    try:
        state, _ = env.reset(seed=seed)
        if render_mode == "ansi":
            print(env.render())
        for steps in range(1, config.max_steps + 1):
            state, reward, terminated, truncated, _ = env.step(int(q_table[state].argmax()))
            total_reward += reward
            if render_mode == "ansi":
                print(env.render())
            if terminated or truncated:
                return {"timesteps": steps, "reward": total_reward,
                        "sucesso": terminated, "truncado": truncated}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="resultados")
    parser.add_argument("--demo", type=Path, help="Q-table .npy para uma partida em texto")
    parser.add_argument("--scenario", choices=SCENARIOS, default="deterministico")
    args = parser.parse_args()
    if args.demo:
        print(demonstrate(args.demo, args.scenario))
    else:
        frames = run_experiments(Config(episodes=args.episodes, runs=args.runs), args.output)
        print(frames["resumo_avaliacoes"].to_string(index=False))


if __name__ == "__main__":
    main()
