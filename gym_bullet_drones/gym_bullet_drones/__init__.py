from gym.envs.registration import register

register(
    id='bullet-drones-v0',
    entry_point='gym_bullet_drones.envs:BulletDronesEnv'
)