# Design Choices

- **No reset function: the environment resets itself**.
    - In `gym.Env`, running the environment after `done=True` but without resetting led to undefined behavior.
    - Another annoyance of `gym.Env` and `VecEnv` was that given an environment, there was no way to get the first observation, which the first action would depend on, without resetting the environment.
    - It's perfectly valid to define an environment without a reset function, and general RL algorithms don't need to use a reset function. Hence, it makes sense to leave it out of the required interface. You're welcome to define a reset function, possibly with arguments, if you plan to pair your environment with an algorithm that uses this feature.
    - *What if I want to reset a whole wrapper stack, which contains stateful wrappers like FrameSkip?* You can use a constructor function to rebuild the object. If the construction process is expensive, you can make your constructor return a fixed object.
- **Vectorized environments are first-class**. We found ourselves using vectorized environments more often than not, because batching necessary for efficient GPU code. However, we ended up with a lot of duplication of utilities between vectorized and non-vectorized environments. To avoid this duplication, we just merge the `n=1` and `n>1` cases. `n=1` is a common special case, and it's assumed that some code will only work for `n=1`, and many envs will be implemented for `n=1` and batched later.
- **Outputs are reordered to `(reward, ob, first)`** for the following reasons
    - This ordering emphasizes that under standard indexing, reward is from previous timestep, whereas first and ob are from current timestep, helping to avoid common off-by-one errors
    - Another common bug is to mix up first and reward since they're both of the same shape. The bug is less likely with this reordering, with observation between them.
- **Use a better type system for spaces**. Using `gym.spaces` leads to a lot of redundancy in dealing with `Discrete` and `MultiDiscrete`, and `Binary` and `MultiBinary`, etc. And what if you need a 2D array of integers? Instead of spaces, we use `ValType`, described as follows.
    ```
    ValType = Dict[str -> ValType] | TensorType
    ```
    I.e., a value is a tensor or a nested dict of tensors.
There's a scalar type `Discrete(k)`, meaning the set `{0, 1, 2, ..., k-1}`, and you can make an n-dimensional tensor of this type. For example, you can use `Discrete(256)` for pixel data.
    - For vectorized `info` dicts, use `get_info()` which will contain a list of dictionaries, 1 per environment.
- The `gym3` module should just provide the basic interface for the environment and wrappers. It doesn't need to contain environments or a make function. This functionality can be provided by other modules.
- **No render function**. The way rendering was defined in `gym` made it much harder to write envs. Each env had to define both how to return an RGB and to pop up a window for the user in an OS-agnostic way. The system was incompatible with environments like games that had their own built-in rendering capabilities. `gym3` is less proscriptive about how rendering should work. You can choose one of the following options depending on your use-case:
    - Construct the environment with some option, such as `render=True`, so that the environment can render all frames to the user. (This makes sense for games or for simulators with built-in rendering.)
    - Include an RGB array in an info dict. Then apply a wrapper that pops up a window showing this array, e.g. `gym3.ViewerWrapper`


## Multi-Agent Usage

`gym3.Env` is designed with multi-agent RL in mind. However, it turns out that this interface does not need to be aware of whether the observation-action streams are completely independent or if they're from different agents in the same world. Here are some ways you can describe a multi-agent system with a `gym3.Env`.

- In the simplest case, an `Env` corresponds to a single multiplayer game with env.num players. If we run an RL algorithm on this `Env`, we’re doing self-play without historical opponents.
    - This setup can be straightforwardly extended to having K concurrent games with N players each, with `env.num == N*K`.
- We can also use a client-server design, where there are multiple client envs that are actually just "proxy" objects that connect to the same world (server). Some matchmaking algorithm decides which agents are to be placed in the same world, possibly including past versions of the agents.
    - Here’s how this would look with separate processes for optimization and rollouts.
        - Version 1 (possibly slower): each rollout process has one (vectorized) Env (client) and one model. When we call env.act, the action get sent to the server which lives in a different process. The trajectory is built up on the rollout process, and when it’s complete, it gets sent to the optimizer.
        - Version 2 (possibly faster): each rollout process has a list of envs and a list of policies. Each timestep, it calls all of the envs and policies in order. The envs are actually just wrappers on an object that waits until all of the actions are provided, then it actually sends the action list to the server. When the trajectory is done, it gets sent to the optimizer. Compared to version 1, this version has less inter-process communication. 
    - Here’s how this would look with the with the rollout in the same process as the optimization. Each training process looks like normal single-agent training, however, actions are being sent to a world server that lives in a different process, which is talking to multiple training processes, and possibly some non-training processes (with historical policies).
- A multiagent rollout function/class as described in Version 2 above can be used for testing the multi-agent environment in a single process (but without RL training).

What if there are different types of agents, with different observations and action spaces? This can be handled by the client-server design: there are multiple client Envs talking to the same world. Each Env only talks to players with the same observation and action spaces.

## Q & A

### How about turn-based games?
What if only one player moves at a time? This is a pretty rare case, so we can make it slightly inconvenient: the non-active player still needs to provide a dummy action, but it's ignored.

### How about adding and removing players?

That's not supported by this interface--you'll need to write some code to batch the data into fixed-sized arrays.
An interface is only useful if you can write generic functions that can handle any object conforming to that interface. It's hard to write generic code that deals with variable numbers of players, so by making our interface include that case, we're making it harder to write library functions that operate on any Env.