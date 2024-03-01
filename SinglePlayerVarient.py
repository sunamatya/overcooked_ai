import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
from overcooked_ai_py.agents.agent import Agent, AgentPair,StayAgent, RandomAgent, GreedyHumanModel
from overcooked_ai_py.overcooked_mdp import OvercookedGridWorld

mdp_gen_params = {"layout_name": 'five_by_five'}
mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
env_params = {"horizon": 100}
agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)

class CustomQAgent(Agent):
    """An agent randomly picks motion actions.
    Note: Does not perform interat actions, unless specified"""

    def __init__(self, mlam):
        #check state space
        mlam= mlam
        self.mdp = self.mlam.mdp
        #check q table shape and the matrix shape
        self.Q_table = np.zeros((self.mdp.shape, Action.NUM_ACTIONS))
        self.exploration_proba = 1
        self.exploration_decreasing_decay = 0.001
        self.min_exploration_proba = 0.01
        self.gamma = 0.9
        self.lr = 0.1

    def action(self, state):
        # return action to maximum Q table in setup
        current_state = state.playerState.get_position()
        if np.random.uniform(0,1) < self.exploration_proba:
            action_probs = np.zeros(Action.NUM_ACTIONS)
            legal_actions = Action.ALL_ACTIONS
            legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
            action_probs[legal_actions_indices] = 1 / len(legal_actions)
            return Action.sample(action_probs), {"action_probs": action_probs}

        else:
            action = np.argmax(self.Q_table[current_state,:])
            action_probs = 0.5 #check what the action probs is supposed to be
            return action, {"action_probs": action_probs}


    def actions(self, states, agent_indices):
        return (self.action(state) for state in states)


# agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())

single_q_agent_pair= AgentPair(CustomQAgent(agent_eval.env.mlam), StayAgent())
trajectories_single_greedy_agent = agent_eval.evaluate_agent_pair(single_q_agent_pair, num_games=1)
print("Random pair rewards", trajectories_single_greedy_agent["ep_returns"])