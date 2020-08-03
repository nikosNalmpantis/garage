import pytest

from garage.envs import MetaWorldSetTaskEnv


@pytest.mark.mujoco
def test_sample_and_step():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    ml1 = metaworld.ML1('push-v1')
    env = MetaWorldSetTaskEnv(ml1, 'train')
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    env.step(env.action_space.sample())
    env.close()
    env2 = MetaWorldSetTaskEnv()
    env2.set_task(task)
    env2.step(env.action_space.sample())
    env2.close()
