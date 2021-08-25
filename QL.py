import numpy as np
import random

num_episodes = 10000
max_steps_per_episodes = 20
learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01


# Define the actions
actions = [0,1,2,3,4,5,6,7,8]

# Define the Moon and Sun position at start
Moon_Sun_Start = {
    '0' : '*',
    '1' : '*',
    '2' : '*',
    '3' : '/',
    '4' : '*',
    '5' : '/',
    '6' : '/',
    '7' : ' ',
    '8' : '/'
}

# Define the Moon and Sun position at the end
Moon_Sun_End = {
    '0' : ' ',
    '1' : '/',
    '2' : '*',
    '3' : '/',
    '4' : '*',
    '5' : '/',
    '6' : '*',
    '7' : '/',
    '8' : '*'
}

# Define proper moves
moves = np.array([[0,1,0,1,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,1,0,0,0],
              [1,0,0,0,1,0,1,0,0],
              [0,1,0,1,0,1,0,1,0],
              [0,0,1,0,1,0,0,0,1],
              [0,0,0,1,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,1,0,1,0]])


rewards_all_episodes = []
q_table = np.array(np.zeros([9,4]))

# Reward function
def situation(state, new_state, MN):
    global Moon_Sun_End
    MN[str(state)], MN[str(new_state)] = MN[str(new_state)], MN[str(state)]
    point = 0
    for i in range(9):
        if MN[str(i)] == Moon_Sun_End[str(i)]:
            point += 1
    if point == 9:
        return point, True
    return 0, False


# Q-Learning
for episode in range(num_episodes):
    MN = Moon_Sun_Start.copy()
    route = [7]
    state = 7
    done = False
    reward_current_episode = 0
    for step in range(max_steps_per_episodes):
        
        # Exploration-Exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate and np.argmax(q_table[state,:]) != 0:
            action = np.argmax(q_table[state,])
            if action == 0:
                new_state = state + 1
            elif action == 1:
                new_state = state + 3
            elif action == 2:
                new_state = state - 1
            else:
                new_state = state - 3
        else:
            playable_actions = []
            for j in range(9):
                if moves[state, j] > 0:
                    playable_actions.append(j)
            new_state = np.random.choice(playable_actions)
            while(len(route) >= 2 and new_state == route[-2]):
                new_state = np.random.choice(playable_actions)
        
        # Computing reward and Situation
        reward, done = situation(state, new_state, MN)
        
        # Updating Q table based on action and reward
        if new_state > state:
            if state + 1 == new_state:
                action = 0 #right
            else:
                action = 1 #down
        else:
            if state - 1 == new_state:
                action = 2 #left
            else:
                action = 3 #up
        q_table[state, action] = q_table[state, action] * \
            (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
       # Upadting route and checking if we reached the goal or not
        route.append(new_state)
        state = new_state
        reward_current_episode += reward
        if done == True:
            print("\n YEEEEEEEEES! ")
            print(route)
            break
        

        
    # Updating exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate * episode)
        
    rewards_all_episodes.append(reward_current_episode)
    


# Results    
print(" \n", q_table)