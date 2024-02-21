import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import Agent, AgentPair, StayAgent, RandomAgent, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from IPython.display import display, Image

# mdp_gen_params = {"layout_name": 'cramped_room'}
# mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
# env_params = {"horizon": 1000}
# agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)
#
# trajectory_random_pair= agent_eval.evaluate_random_pair(num_games=5)
# print("Random pair rewards", trajectory_random_pair["ep_returns"])

mdp_gen_params = {"inner_shape": (7,7),
                 "prop_empty": 0.2, #proportion of empty space
                 "prop_feats": 0.8, #proportion of occupied space or counters with features in them
                 "display": True,
                 "start_all_orders":
                 [{"ingredients":["onion", "onion", "onion"]},
                 {"ingredients":["onion", "onion"]},
                 {"ingredients":["onion"]}],
                 "recipie_values": [20, 9, 4],
                 "recipie_times":[20,15,10]
                 }

env_params= {"horizon": 500}

#need to check why the outer_shape is not in the original mpd_gen_params
mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape= (7,7))
agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)

#
# #need to check why the outer_shape is not in the original mpd_gen_params
mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params, outer_shape= (7,7))
agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)

trajectories_random_pair = agent_eval.evaluate_random_pair(num_games=1)
print("Random pair rewards", trajectories_random_pair["ep_returns"])

def pretty_grid(grid):
    return "\n".join("".join(line) for line in grid)

print("\nGenerated Grid:\n"+ pretty_grid(trajectories_random_pair["mdp_params"][0]["terrain"]))
print("random pair trajectory on generated grid")
#StateVisualizer().display_rendered_trajectory(trajectories_random_pair, trajectory_idx=0, ipython_display=True)
#StateVisualizer().display_rendered_trajectory(trajectories_random_pair, ipython_display=True)
grid = agent_eval.env.mdp.terrain_mtx
starting_state_random_pair= trajectories_random_pair["ep_states"][0][0]
print("Staring state of random pair")

img_path = StateVisualizer().display_rendered_state(starting_state_random_pair, grid=grid, ipython_display=True)
display(img_path)

# class CustomRandomAgent(Agent):
#     """An agent randomly picks motion actions.
#     Note: Does not perform interat actions, unless specified"""
#
#     def action(self, state):
#         action_probs = np.zeros(Action.NUM_ACTIONS)
#         legal_actions = Action.ALL_ACTIONS
#         legal_actions_indices = np.array([Action.ACTION_TO_INDEX[motion_a] for motion_a in legal_actions])
#         action_probs[legal_actions_indices] = 1 / len(legal_actions)
#         return Action.sample(action_probs), {"action_probs": action_probs}
#
#     def actions(self, states, agent_indices):
#         return (self.action(state) for state in states)
#
#
# agent_pair = AgentPair(CustomRandomAgent(), CustomRandomAgent())
# mdp_gen_params = {"layout_name": 'cramped_room'}
# mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
# env_params = {"horizon": 1000}
# agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)
#
# # check the choice of the agent pair
# trajectory_custom_random_pair = agent_eval.evaluate_agent_pair(agent_pair, num_games=10)
# print("Custom random pair rewards", trajectory_custom_random_pair["ep_returns"])