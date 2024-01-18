import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math


man_weather = pd.read_csv("./manchester_weather.csv") # mid, rainy
ban_weather = pd.read_csv("./bangkok_weather.csv") # hot
yel_weather = pd.read_csv("./yellowknife_weather.csv") # cold


def get_weight(temp, humidity, precip):
    #warmth, breathability, comfort, waterproof, stretch, moisturewick
    weight = [0, 0, 0, 0, 0, 0]
    a = 0.15
    weight[0] = math.tanh(2-(temp*a)) #warmth weighting, relates to temp

    #breathability weighting, relates to temp, humidity added as an extra heuristic
    a1 = 0.065
    weight[1] = a1*max(0, temp) + math.exp(0.01*humidity) - 2 
    
    #Comfort, relates to temp levels. More comfort assumed to be needed for levels at extremes
    a2=0.01
    weight[2] = a2*((temp - 15)**2)

    weight[3] = 0.1 * precip #higher precipitation  = more waterproof

    weight[4] = weight[2] #assumed that stretch correlates with level of comfort needed

    weight[5] = humidity * 0.01 #humidity levels calculated according to percentage

    for i in range(len(weight)):
        if(weight[i] > 1):
            weight[i] = 1
        if(weight[i] < -1):
            weight[i] = -1

    return weight

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def q_learning(weighted_score):

    material_scores = pd.read_csv("./material_scores-1.csv")
    material_scores = material_scores.set_index("Material")
    clothes_data = pd.read_csv("./data.csv")


    pass

    name = []
    reward = []
    max_reward = []
    reward_difference = []
    abs_scores = []




# BEST SCORE POSSIBLE - 0.7*1+1+0.2+0.6+0+0.8 = 3.3
# OLD
# 0.5 * warm + 7 * breathability + -1 * comfort
# --------

# https://stackoverflow.com/questions/19989919/q-learning-multiple-goals#:~:text=The%20simplest%20way%20is%20to,to%20make%20a%20total%20reward.&text=you%20can%20decide%20then%20how,tries%20to%20learn%20something%20different.
# Multi Dimensional Reward Reinforcement Learning

    for i,row in clothes_data.iterrows():
        name.append(row["Title"])
        reward.append(((material_scores.loc[row["Material A"]] * row["A Percent"] +  material_scores.loc[row["Material B"]] * row["B Percent"]) * weighted_score).sum())  
        max_reward.append(np.abs(np.array(weighted_score)).sum())
        reward_difference.append( (np.abs(np.array(weighted_score)).sum()) -
            (((material_scores.loc[row["Material A"]] * row["A Percent"] +  material_scores.loc[row["Material B"]] * row["B Percent"]) * weighted_score).sum()))
 
        abs_scores.append(np.abs(np.array(weighted_score)))
    df = pd.DataFrame({
        "name": name,
        "reward": reward,
        "max possible reward": max_reward,
        "optimum-real reward difference":reward_difference,
        "absolute scores": abs_scores

    })


    num_actions = 2
    num_observations = df.shape[0]
    q_score = np.zeros((num_observations,num_actions))

    learning_rate = 0.01
    discount_factor = 0.99
    num_episodes = 10000
    # Number of episodes the bigger the no epsiodes exploitation, randomness is for exploration
    max_iter_episode = 50
    # number of iterations more then more accurate the Q value is
    exploration_prob = 1
    exploration_decreasing_decay = 0.001
    #minimum of exploration proba
    min_exploration_proba = 0.01

    # Q learning is action based - it calculates the best path, only goes up and down through database
    actions = [-1,1]

    rewards_per_episode = list()


    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=False)
    choose_history = []
    learning_history = []


    for e in range(num_episodes):
        print("----EPISODE " + str(e) + "----",end="\r")
        current_state = np.random.randint(0,num_observations-1)
        total_episode_reward = 0
        for i in range(max_iter_episode):
            if(current_state < 0):
                current_state = 0
            if(current_state >= num_observations):
                current_state = num_observations - 1
        
            if(np.random.uniform(0,1) < exploration_prob):
                action = np.random.randint(0,2)
            else:
                action = np.argmax(q_score[current_state])


            current_action = actions[action]
            next_state = current_state + current_action

            if(next_state >= num_observations):
                next_state = 0
            if(current_state + current_action < 0):
                next_state = num_observations - 1

         # ^^ This is edge detection. If the postion plus the action goes outside of the array length, it wraps around
            reward = df.loc[current_state]["reward"]
            q_score[current_state][action] = (1-learning_rate) * q_score[current_state][action] + (learning_rate)* (reward + discount_factor *  max(q_score[next_state]))

            total_episode_reward = total_episode_reward + reward
            current_state = next_state
        
        exploration_prob = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))

        rewards_per_episode.append(total_episode_reward)
        choose_history.append(q_score.argmax(axis=0)[0])


 

    #LABEL ALL THESE PLOTS!!!!!!!
    # reward_average = moving_average(rewards_per_episode,250)
    # plt.bar(df.index,q_score.T[0])
    # plt.xlabel('Product Index')
    # plt.ylabel('Q Score Values')
    # plt.title('Q Score values for each Item')
    # plt.show()

    # plt.plot(rewards_per_episode)
    # plt.title('Rewards per Episode')
    # plt.xlabel('Number of Episodes')
    # plt.ylabel('Cumulative Episode Reward')
    # plt.plot(reward_average)
    # plt.show()
    
    # plt.plot(choose_history)
    # plt.title('Convergence Plot')
    # plt.xlabel('Episodes')
    # plt.ylabel("Product Index")
    # plt.show()




    return(q_score.max(axis=0), df.loc[q_score.argmax(axis=0)[0]])


    

def main():
    sample_size = 2
    man_df = man_weather.sample(sample_size) #mid, rainy
    ban_df = ban_weather.sample(sample_size) #hot
    yel_df = yel_weather.sample(sample_size) #cold

#TESTING -------------------------------------------------------------------------
    man_current_weather = get_weight(9.2, 75.1, 0.317)
    print("=====================================================================")
    print("MANCHESTER CURRENT WEIGHTS: \n",man_current_weather)
    score, clothing = q_learning(man_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)
    
    man_current_weather = get_weight(12.3, 84.9, 0.0)
    print("=====================================================================")
    print("MANCHESTER CURRENT WEIGHTS: \n",man_current_weather)
    score, clothing = q_learning(man_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    man_current_weather = get_weight(10.1, 82.7, 0.0)
    print("=====================================================================")
    print("MANCHESTER CURRENT WEIGHTS: \n",man_current_weather)
    score, clothing = q_learning(man_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    man_current_weather = get_weight(17.5, 79.6, 0.0)
    print("=====================================================================")
    print("MANCHESTER CURRENT WEIGHTS: \n",man_current_weather)
    score, clothing = q_learning(man_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    man_current_weather = get_weight(5.8, 88.6, 0.812)
    print("=====================================================================")
    print("MANCHESTER CURRENT WEIGHTS: \n",man_current_weather)
    score, clothing = q_learning(man_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)


    ban_current_weather = get_weight(27.8, 54.8, 0.0)
    print("=====================================================================")
    print("BANGKOK CURRENT WEIGHTS: \n",ban_current_weather)
    score, clothing = q_learning(ban_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    ban_current_weather = get_weight(27.5, 59.7, 0.0)
    print("=====================================================================")
    print("BANGKOK CURRENT WEIGHTS: \n",ban_current_weather)
    score, clothing = q_learning(ban_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    ban_current_weather = get_weight(30.3, 69.1, 0.0)
    print("=====================================================================")
    print("BANGKOK CURRENT WEIGHTS: \n",ban_current_weather)
    score, clothing = q_learning(ban_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    ban_current_weather = get_weight(28.3, 84.2, 34.237)
    print("=====================================================================")
    print("BANGKOK CURRENT WEIGHTS: \n",ban_current_weather)
    score, clothing = q_learning(ban_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    ban_current_weather = get_weight(29.8, 72.9, 0.0)
    print("=====================================================================")
    print("BANGKOK CURRENT WEIGHTS: \n",ban_current_weather)
    score, clothing = q_learning(ban_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)


    yel_current_weather = get_weight(16.4, 59.0, 0.0)
    print("=====================================================================")
    print("YELLOWKNIFE CURRENT WEIGHTS: \n",yel_current_weather)
    score, clothing = q_learning(yel_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score) 

    yel_current_weather = get_weight(15.2, 64.8, 0.0)
    print("=====================================================================")
    print("YELLOWKNIFE CURRENT WEIGHTS: \n",yel_current_weather)
    score, clothing = q_learning(yel_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    yel_current_weather = get_weight(13.1, 88.9, 4.0)
    print("=====================================================================")
    print("YELLOWKNIFE CURRENT WEIGHTS: \n",yel_current_weather)
    score, clothing = q_learning(yel_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)


    yel_current_weather = get_weight(-17.5, 84.2, 0.0)
    print("=====================================================================")
    print("YELLOWKNIFE CURRENT WEIGHTS: \n",yel_current_weather)
    score, clothing = q_learning(yel_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

    yel_current_weather = get_weight(-15.5, 84.7, 0.0)
    print("=====================================================================")
    print("YELLOWKNIFE CURRENT WEIGHTS: \n",yel_current_weather)
    score, clothing = q_learning(yel_current_weather)
    print("CLOTHING: ", clothing)
    print("SCORE: ", score)

#----------------------------------------------------------------------------------

    # for i,row in man_df.iterrows():
    #    man_current_weather = get_weight(row["temp"], row["humidity"], row["precip"])
    #    print("=====================================================================")
    #    print("MANCHESTER CURRENT SAMPLED WEATHER [",i,"]: \n", row)
    #    print("MANCHESTER CURRENT WEIGHTS [",i,"]: \n",man_current_weather)
       
    #    score, clothing = q_learning(man_current_weather)
    #    print("CLOTHING: ", clothing)
    #    print("SCORE: ", score)

    # for i,row in ban_df.iterrows():
    #    ban_current_weather = get_weight(row["temp"], row["humidity"], row["precip"])
    #    print("=====================================================================")
    #    print("BANGKOK CURRENT SAMPLED WEATHER [",i,"]: \n", row)
    #    print("BANGKOK CURRENT WEIGHTS [",i,"]: \n",ban_current_weather)
       
    #    score, clothing = q_learning(ban_current_weather)
    #    print("CLOTHING: ", clothing)
    #    print("SCORE: ", score)

    # for i,row in yel_df.iterrows():
    #    yel_current_weather = get_weight(row["temp"], row["humidity"], row["precip"])
    #    print("=====================================================================")
    #    print("YELLOWKNIFE CURRENT SAMPLED WEATHER [",i,"]: \n", row)
    #    print("YELLOWKNIFE CURRENT WEIGHTS [",i,"]: \n", yel_current_weather)
       
    #    score, clothing = q_learning(yel_current_weather)
    #    print("CLOTHING: ", clothing)
    #    print("SCORE: ", score)


if __name__ == "__main__":
    main()