import gridworld
import numpy as np

def bellman(env, Q, s, a, gamma):
    s_, r, done = env.dynamics(s, a)
    if done:
        Q[s, a] = r
    else:
        Q[s, a] = r + gamma * np.max(Q[s_])


def learn_q(env, gamma):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    while 1:
        Q_ = Q.copy()
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                bellman(env, Q, s, a, gamma)
        if (Q_ == Q).all():
            break
    return Q

def learn_policy(env, gamma):
    return learn_q(env, gamma).argmax(-1)


if __name__ == '__main__':
    board = np.zeros([8, 8])
    board[0, -1] = 1

    env = gridworld.Grid(board=board, start_orientation=0, start_pos=np.zeros(2), goal_pos=np.ones(2) * 7)
    pi = learn_policy(env, 0.99)

    state = env.reset()
    env.render()
    for _ in range(100):
        state, _, done, _ = env.step(pi[state])
        env.render()
        if done:
            break
                
