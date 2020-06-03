"""
Example random agent script using gym3
"""

import argparse
import gym3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn_path", help="python path to the environment class", required=True
    )
    parser.add_argument(
        "--render_mode",
        help="the render_mode to pass to the environment, if set, should be one of `rgb_array` or `human`",
    )
    args, remaining_args = parser.parse_known_args()

    # convert types for extra args
    env_kwargs = {}
    while len(remaining_args) > 0:
        arg = remaining_args.pop(0)
        if "=" in arg:
            key, _sep, value = arg.partition("=")
        else:
            key = arg
            value = remaining_args.pop(0)
        assert key.startswith("--")
        for prefix, convert in [
            ("int", int),
            ("bool", lambda s: s.lower() in ["true", "t"]),
        ]:
            if value.startswith(prefix + ":"):
                value = convert(arg[len(prefix) + 1])
                break
        env_kwargs[key[2:]] = value

    # instantiate environment
    env = gym3.call_func(
        args.fn_path, num=1, render_mode=args.render_mode, **env_kwargs
    )
    if args.render_mode == "rgb_array":
        env = gym3.ViewerWrapper(env, info_key="rgb")

    # take random actions in environment
    step = 0
    while True:
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
        print(f"step {step} reward {rew} first {first}")
        step += 1


if __name__ == "__main__":
    main()
