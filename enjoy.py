import collections
import os
import pickle
import time

import gym
import ray
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.registry import get_agent_class
# noinspection PyProtectedMember
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import TupleActions
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.rollout import create_parser, DefaultMapping
from ray.tune.util import merge_dicts

from algorithms.models.vizdoom_model import VizdoomVisionNetwork
from envs.doom.doom_utils import register_doom_envs_rllib, DEFAULT_FRAMESKIP
from utils.utils import log


def create_parser_custom():
    parser = create_parser()
    parser.add_argument(
        '--dbg',
        action='store_true',
        help='Full debug mode (also enables local-mode)',
    )
    return parser


def run(args, parser):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    local_mode = False
    if args.dbg:
        local_mode = True
        config['num_workers'] = 1
        config['num_gpus'] = 1
        config['num_envs_per_worker'] = 1

    ray.init(local_mode=local_mode)

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout_loop(agent, args.env, num_steps, args.no_render, fps=45)


# noinspection PyUnusedLocal
def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout_loop(agent, env_name, num_steps, no_render=True, fps=1000):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    last_render_time = time.time()

    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id_: state_init[mapping_cache[agent_id_]])
        prev_actions = DefaultMapping(
            lambda agent_id_: action_init[mapping_cache[agent_id_]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)

                    if isinstance(env.action_space, gym.spaces.Tuple):
                        a_action = TupleActions(a_action)
                        a_action = _unbatch_tuple_actions(a_action)[0]

                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            frameskip = DEFAULT_FRAMESKIP
            rewards = None
            for frame in range(frameskip):
                next_obs, reward, done, _ = env.step(action)
                if rewards is None:
                    rewards = reward
                else:
                    if multiagent:
                        for agent_id, r in reward.items():
                            rewards[agent_id] += r
                    else:
                        rewards += reward

                if not no_render:
                    time_since_last_render = time.time() - last_render_time
                    time_between_frames = 1.0 / fps
                    time_wait = time_between_frames - time_since_last_render
                    if time_wait > 0:
                        time.sleep(time_wait)
                    env.render()
                    last_render_time = time.time()
                steps += 1
                obs = next_obs

            if multiagent:
                for agent_id, r in rewards.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = rewards

            if multiagent:
                done = done['__all__']
                reward_total += sum(rewards.values())
            else:
                reward_total += rewards

        print('Episode reward', reward_total)


def main():
    register_doom_envs_rllib(mode='test')

    ModelCatalog.register_custom_model('vizdoom_vision_model', VizdoomVisionNetwork)

    parser = create_parser_custom()
    args = parser.parse_args()
    run(args, parser)


if __name__ == '__main__':
    main()
