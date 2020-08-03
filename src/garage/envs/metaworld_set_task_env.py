"""Environment that wraps a MetaWorld benchmark in the set_task interface."""
import random

# Importing garage directly avoids circular dependencies with GymEnv,
# TaskNameWrapper, and TaskOnehotWrapper.
import garage
from garage import Environment


class MetaWorldSetTaskEnv(Environment):
    """Environment form of a MetaWorld benchmark.

    This class is generally less efficient than using a TaskSampler, if that
    can be used instead, since each instance of this class internally caches a
    copy of each environment in the benchmark.

    In order to sample tasks from this environment, a benchmark must be passed
    at construction time.

    Args:
        benchmark (metaworld.Benchmark or None): The benchmark to wrap.
        kind (str or None): Whether to use test or train tasks.
        wrapper (Callable[garage.Env, garage.Env] or None): Wrapper to apply to
            env instances.
        add_env_onehot (bool): If true, a one-hot representing the current
            environment name will be added to the environments. Should only be
            used with multi-task benchmarks.

    Raises:
        ValueError: If kind is not 'train', 'test', or None. Also raisd if
            `add_env_onehot` is used on a metaworld meta learning (not
            multi-task) benchmark.

    """

    def __init__(self,
                 benchmark=None,
                 kind=None,
                 wrapper=None,
                 add_env_onehot=False):
        self._constructed_from_benchmark = benchmark is not None
        if self._constructed_from_benchmark:
            assert kind is not None
        else:
            assert kind is None
            assert wrapper is None
            assert add_env_onehot is False
        self._benchmark = benchmark
        self._kind = kind
        self._wrapper = wrapper
        self._add_env_onehot = add_env_onehot
        self._envs = {}
        self._current_task = None
        self._inner_tasks = []
        self._classes = {}
        self._tasks = []
        self._next_task = 0
        if self._benchmark is not None:
            self._fill_tasks()
            self.set_task(self._tasks[0])
            self.reset()

    @property
    def num_tasks(self):
        """int: Returns number of tasks.

        Part of the set_task environment protocol.

        """
        assert self._benchmark is not None
        return len(self._tasks)

    def sample_tasks(self, n_tasks):
        """Samples n_tasks tasks.

        Part of the set_task environment protocol. To call this method, a
        benchmark must have been passed in at environment construction.

        Args:
            n_tasks (int): Number of tasks to sample.

        Returns:
            dict[str,object]: Task object to pass back to `set_task`.

        """
        assert self._constructed_from_benchmark
        tasks = []
        while len(tasks) < n_tasks:
            if self._next_task == len(self._tasks):
                random.shuffle(self._tasks)
                self._next_task = 0
            tasks.append(self._tasks[self._next_task])
            self._next_task += 1
        return tasks

    def set_task(self, task):
        """Set the task.

        Part of the set_task environment protocol.

        Args:
            task (dict[str,object]): Task object from `sample_tasks`.

        """
        # Mixing train and test is probably a mistake
        assert self._kind is None or self._kind == task['kind']
        self._benchmark = task['benchmark']
        self._kind = task['kind']
        self._add_env_onehot = task['add_env_onehot']
        if not self._tasks:
            self._fill_tasks()
        self._current_task = task['inner']
        self._construct_env_if_needed()
        self._current_env.set_task(task['inner'])
        self._current_env.reset()

    def _fill_tasks(self):
        """Fill out _tasks after the benchmark is set.

        Raises:
            ValueError: If kind is not set to "train" or "test"
        """
        if self._add_env_onehot:
            if (self._kind == 'test'
                    or 'metaworld.ML' in repr(type(self._benchmark))):
                raise ValueError('add_env_onehot should only be used with '
                                 'multi-task benchmarks, not ' +
                                 repr(self._benchmark))
            self._task_indices = {
                env_name: index
                for (index, env_name) in enumerate(self._classes.keys())
            }
        self._tasks = []
        if self._kind is None:
            return
        if self._kind == 'test':
            self._inner_tasks = self._benchmark.test_tasks
            self._classes = self._benchmark.test_classes
        elif self._kind == 'train':
            self._inner_tasks = self._benchmark.train_tasks
            self._classes = self._benchmark.train_classes
        else:
            raise ValueError('kind should be either "train" or "test", not ' +
                             repr(self._kind))
        for task in self._inner_tasks:
            self._tasks.append({
                'kind': self._kind,
                'benchmark': self._benchmark,
                'add_env_onehot': self._add_env_onehot,
                'inner': task,
            })

    def _construct_env_if_needed(self):
        """Construct current_env if it doesn't exist."""
        env_name = self._current_task.env_name
        env = self._envs.get(env_name, None)
        if env is None:
            env = self._classes[env_name]()
            env.set_task(self._current_task)
            env = garage.envs.GymEnv(env,
                                     max_episode_length=env.max_path_length)
            env = garage.envs.TaskNameWrapper(env, task_name=env_name)
            if self._add_env_onehot:
                task_index = self._task_indices[env_name]
                env = garage.envs.TaskOnehotWrapper(env,
                                                    task_index=task_index,
                                                    n_total_tasks=len(
                                                        self._task_indices))
            if self._wrapper is not None:
                env = self._wrapper(env, self._current_task)
            self._envs[env_name] = env

    @property
    def _current_env(self):
        """garage.Environment: The current environment."""
        assert self._current_task is not None
        return self._envs[self._current_task.env_name]

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._current_env.action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._current_env.observation_space

    @property
    def spec(self):
        """EnvSpec: The envionrment specification."""
        return self._current_env.spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._current_env.render_modes

    def step(self, action):
        """Step the wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        """
        return self._current_env.step(action)

    def reset(self):
        """Reset the wrapped env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        return self._current_env.reset()

    def render(self, mode):
        """Render the wrapped environment.

        Args:
            mode (str): the mode to render with. The string must be
                present in `self.render_modes`.

        Returns:
            object: the return value for render, depending on each env.

        """
        return self._current_env.render(mode)

    def visualize(self):
        """Creates a visualization of the wrapped environment."""
        self._current_env.visualize()

    def close(self):
        """Close the wrapped env."""
        for env in self._envs.values():
            env.close()
