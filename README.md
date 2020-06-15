**Status:** Maintenance (expect bug fixes and minor updates)

# gym3

`gym3` provides a unified interface for reinforcement learning environments that improves upon the `gym` interface and includes vectorization, which is invaluable for performance.  `gym3` is just the interface and associated tools, and includes no environments beyond some simple testing environments.

`gym3` is used internally inside OpenAI and is released here primarily for use by OpenAI environments.  External users should likely use [`gym`](https://github.com/openai/gym).

Supported platforms:

- Windows
- macOS
- Linux

Supported Pythons:

- `>=3.6`

Installation:

`pip install gym3`

## Overview

`gym3.Env` is similar to combining multiple `gym.Env` environments into a single environment, with automatic reset when episodes are complete.

A `gym3` random agent looks like this (run `pip install --upgrade procgen` to get the environment):

```py
import gym3
from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=2, env_name="coinrun")
step = 0
while True:
    env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
```

To visualize what the agent is doing:

```py
import gym3
from procgen import ProcgenGym3Env
env = ProcgenGym3Env(num=2, env_name="coinrun", render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")
step = 0
while True:
    env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
```

A command line example is included under `scripts`:

```
python -m gym3.scripts.random_agent --fn_path procgen:ProcgenGym3Env --env_name coinrun --render_mode rgb_array
```

The observations and actions can be either arrays, or "trees" of arrays, where a tree is a (potentially nested) dictionary with string keys.  `gym3` includes a handy function, `gym3.types.multimap` for mapping functions over trees, as well as a number of utilities in `gym3.types_np` that produce trees numpy arrays from space objects, such as `types_np.sample()` seen above.

Compatibility with existing `gym` environments is provided as well:

```py
import gym3
env = gym3.vectorize_gym(num=2, render_mode="human", env_kwargs={"id": "CartPole-v0"})
step = 0
while True:
    env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
    rew, obs, first = env.observe()
    print(f"step {step} reward {rew} first {first}")
    step += 1
```

## Documentation

* [API Reference](docs/api.md)
* [`gym3` for `gym` users](docs/gym3-for-gym-users.md)
* [Design Choices](docs/design.md)

## Example Environments

* [Example Gridworld](https://github.com/christopher-hesse/example-gridworld) - Simple gridworld where you observe the state directly
* [Computer Tennis](https://github.com/christopher-hesse/computer-tennis) - Clone of the game "Pong" where you observe pixels. Renders with Cairo or OpenGL headless (no X server required)
* [`libenv_fixedseq.c`](gym3/libenv_fixedseq.c) - Example environment using the `libenv` C API.  For a full C++ environment that uses this interface, see [Procgen Benchmark](https://github.com/openai/procgen)

## Changelog

See [CHANGES](CHANGES.md) for changes present in each release.

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for information on contributing.