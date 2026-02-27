import numpy as np
from src.chefshat_env.agents.random_agent import RandomAgent
from src.rl_agents.dqn_agent import DQNAgent


class ChefsHatDQNAgent(RandomAgent):
    """
    DQN agent built ON TOP of RandomAgent
    (required for Chef's Hat to actually request actions)
    """

    def __init__(self, name, log_directory, training=True):
        super().__init__(name=name, log_directory=log_directory)

        self.training = training

        self.state_dim = 20
        self.action_dim = 200

        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        self.prev_state = None
        self.prev_action = None

    def build_state(self, payload):
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = payload.get("round", 0)
        state[1] = payload.get("current_player_index", 0)
        return state

    async def on_action_request(self, payload):
        actions = payload.get("actions", [])

        if not actions:
            return await super().on_action_request(payload)

        state = self.build_state(payload)

        action_idx = self.agent.select_action(state, actions)
        action_idx = int(np.clip(action_idx, 0, len(actions) - 1))

        self.prev_state = state
        self.prev_action = action_idx

        return {
            "action": actions[action_idx],
            "valid_action": True
        }

    async def on_game_update(self, payload):
        if not self.training:
            return

        if payload.get("type") == "match_over":
            scores = payload.get("scores", {})
            reward = scores.get(self.name, 0)

            next_state = np.zeros(self.state_dim, dtype=np.float32)

            self.agent.store_transition(
                self.prev_state,
                self.prev_action,
                reward,
                next_state,
                done=True
            )

            self.agent.train_step()

            self.prev_state = None
            self.prev_action = None