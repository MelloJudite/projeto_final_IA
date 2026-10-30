"""Testes dos alvos TD, da chuva e do protocolo experimental."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from taxi_rl import Config, evaluate, make_env, summarize_evaluations, td_update, train


class TemporalDifferenceTests(unittest.TestCase):
    def test_q_learning_uses_maximum_and_sarsa_uses_chosen_action(self):
        q = np.array([[0., 0.], [2., 10.]])
        s = q.copy()
        td_update(q, 0, 0, -1, 1, False, "q_learning", 0.5, 0.9)
        td_update(s, 0, 0, -1, 1, False, "sarsa", 0.5, 0.9, next_action=0)
        self.assertAlmostEqual(q[0, 0], 4.)
        self.assertAlmostEqual(s[0, 0], 0.4)

    def test_termination_does_not_bootstrap(self):
        for algorithm in ("q_learning", "sarsa"):
            q = np.full((2, 2), 100.)
            target = td_update(q, 0, 0, 20, 1, True, algorithm, 1., 0.99)
            self.assertEqual(target, 20)
            self.assertEqual(q[0, 0], 20)

    def test_sarsa_executes_the_action_used_in_its_target(self):
        class TwoStepEnv:
            observation_space = type("Space", (), {"n": 3})()
            action_space = type("Space", (), {"n": 2})()
            def reset(self, **kwargs):
                self.actions = []
                return 0, {}
            def step(self, action):
                self.actions.append(action)
                return len(self.actions), -1, len(self.actions) == 2, False, {}
            def close(self):
                pass
        env = TwoStepEnv()
        config = Config(episodes=1, runs=1, max_steps=2)
        with patch("taxi_rl.make_env", return_value=env), \
             patch("taxi_rl.epsilon_greedy", side_effect=[1, 0]), \
             patch("taxi_rl.evaluate", return_value=pd.DataFrame(
                 {"reward": [0], "timesteps": [2], "sucesso": [True]})):
            train("sarsa", "deterministico", config, seed=0)
        self.assertEqual(env.actions, [1, 0])

    def test_external_truncation_keeps_bootstrap(self):
        # O estado truncado não é terminal: seu valor deve continuar no alvo.
        class TruncatedEnv:
            observation_space = type("Space", (), {"n": 1})()
            action_space = type("Space", (), {"n": 1})()
            def reset(self, **kwargs):
                return 0, {}
            def step(self, action):
                return 0, -1, False, True, {}
            def close(self):
                pass
        config = Config(episodes=2, runs=1, alpha=1, alpha_decay=0, gamma=0.9)
        for algorithm in ("q_learning", "sarsa"):
            with patch("taxi_rl.make_env", return_value=TruncatedEnv()), \
                 patch("taxi_rl.evaluate", return_value=pd.DataFrame(
                     {"reward": [0], "timesteps": [1], "sucesso": [False]})):
                table, _, _ = train(algorithm, "deterministico", config, 0)
            self.assertAlmostEqual(table[0, 0], -1.9)


class EnvironmentTests(unittest.TestCase):
    def test_rainy_transitions_and_time_limit(self):
        env = make_env("chuva", Config())
        try:
            self.assertEqual(env.spec.id, "Taxi-v4")
            self.assertEqual(env.spec.max_episode_steps, 200)
            # Estado interno usado só para inspecionar a definição da transição.
            transitions = env.unwrapped.P[0][0]
            np.testing.assert_allclose([t[0] for t in transitions], [0.6, 0.2, 0.2])
            self.assertEqual(len(env.unwrapped.P[0][4]), 1)
        finally:
            env.close()

    def test_seeded_training_is_reproducible(self):
        config = Config(episodes=8, runs=1, max_steps=15, validation_episodes=2)
        for algorithm in ("q_learning", "sarsa"):
            first = train(algorithm, "chuva", config, 7)
            second = train(algorithm, "chuva", config, 7)
            np.testing.assert_array_equal(first[0], second[0])
            pd.testing.assert_frame_equal(first[1], second[1])
            pd.testing.assert_frame_equal(first[2], second[2])

    def test_evaluation_is_frozen_and_reports_censoring(self):
        table = np.zeros((500, 6))
        before = table.copy()
        config = Config(evaluation_episodes=3, max_steps=10)
        result = evaluate(table, "deterministico", config)
        np.testing.assert_array_equal(table, before)
        self.assertEqual(len(result), 3)
        self.assertTrue(result.truncado.all())
        self.assertFalse(result.sucesso.any())
        self.assertTrue((result.timesteps == 10).all())

    def test_success_statistics_exclude_truncated_episodes(self):
        frame = pd.DataFrame({"algoritmo": ["sarsa"] * 3,
                              "treino": ["chuva"] * 3, "teste": ["chuva"] * 3,
                              "timesteps": [10, 12, 200], "sucesso": [True, True, False],
                              "truncado": [False, False, True], "reward": [11, 9, -200],
                              "acoes_ilegais": [0, 0, 0]})
        result = summarize_evaluations(frame).iloc[0]
        self.assertEqual(result.n_partidas, 3)
        self.assertAlmostEqual(result.taxa_sucesso, 2 / 3)
        self.assertEqual(result.timesteps_medios_sucesso, 11)
        self.assertAlmostEqual(result.timesteps_dp_sucesso, np.sqrt(2))

    def test_overlapping_test_and_training_seeds_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(Config(), evaluation_seed=2026)


if __name__ == "__main__":
    unittest.main()
