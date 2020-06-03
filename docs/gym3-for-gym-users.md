# `gym3` For `gym` Users

## How do I create a `gym3` environment from a `gym` environment?

You can use `gym3.FromGymEnv` to convert a single instance of an environment.

If you want to assemble a number of environments into a single `gym3` vectorized environment (similar to `baselines.SubprocVecEnv`) try `gym3.vectorize_gym`:

```
import gym3
import gym

env = gym3.vectorize_gym(num=2, env_kwargs={"id": "Pendulum-v0"})
```

## How do I create a `gym3` environment from a string similar to `gym.make`?

`gym3` does not include a registry.  Instead, if you want to create an environment from a string, provide the python path for the environment's constructor:

```
env = gym3.call_func("gym3.testing:IdentityEnv")
```

You can provide keyword arguments to this function that will get passed to the constructor.  If you want to version your environment, you can name it something like `IdentityEnvV2`.

In general we recommend that papers specify the exact version (git commit hash or `==` pypi version number) for their environments to maximize reproducibility.

## How do I render an environment?

The `gym3` interface does not define a standard way to render things, but there are a few conventions that make this more consistent:

* for gym's `mode="rgb_array"`: environments can take a `render_mode="rgb_array"` argument if they support creating images.  The rendered frames should appear under the key `rgb` in the environment's info dictionary.  `gym3.ViewerWrapper` can be used to display these, or agent observations, to the user.
* for gym's `mode="human"`: environments can take a `render_mode="human"` argument, if they support creating their own windows.

In both cases, there is no `env.render()` function to call, and it is up the environment to produce the image during `observe()` or `get_info()`, or else update its own window during `act()` or asynchronously.

The `FromGymEnv` wrapper takes a `render_mode` argument that acts as described above.

## How do I add a custom methods (e.g. `restart()`, `set_state()`)?

You can define the method on your environment object, then call it directly or via `env.callmethod()`.  `env.callmethod()` which works across `SubprocEnv` and `ConcatEnv` but requires that the method arguments and return values all be lists with the length `env.num`.

If you want to handle custom methods in wrappers, you can override `callmethod()` in your wrapper to delegate the call to the wrapper's version and call the original method if desired though this requires that you always use `callmethod()` to access custom methods.  As an example:

```
class RestartableWrapper(gym3.Wrapper):
    def restart(self):
        print("wrapper restart")
        
    def callmethod(self, method, *args, **kwargs):
        if method == "restart":
            self.restart(*args, **kwargs)
        self.env.callmethod(method, *args, **kwargs)
```

## How do I reset, seed, or close an environment?

The `gym3` interface does not define a standard way to do these things.  Environments are free to define their own methods.  See the previous question about adding a custom function.

For the simple case of resetting an environment, you can re-create the environment from scratch.  If this is too slow, either adding a custom method or creating a pool of environments asynchronously can reduce the time.

You can also provide a special action that resets the environment, though you must then make sure your agent cannot execute that action if you don't want that.

## How does this interact with `gym.vector.VectorEnv`?

`gym3` does not currently have an adapter defined for `gym.vector.VectorEnv`, though you could likely apply `gym.vector.VectorEnv` to a `gym3` environment that has been converted to a `gym` environment.

## What about `baselines.VecEnv`?

`gym3` contains a `baselines.VecEnv` adapter `ToBaselinesVecEnv` that can be used to adapt `gym3` environments to work with code expecting `baselines.VecEnv` environments.