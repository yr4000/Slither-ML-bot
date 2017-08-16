import numpy as np
#hyper parameters for reward function
DISCOUNT_FACTOR = 0.85
BACKWARDS_RELEVANCY_WINDOW = 10
BOOSTING_FACTOR = 3.0
PUNISHMENT_FOR_DEATH = -50

#boosts by BOOSTING_FACTOR all the BACKWARDS_RELEVANCY_WINDOW rewards before an
#actual point was earned in the game
def boost(relevancy_indicator , processed_rewards):
    for i in (range(len(relevancy_indicator))):
        if(relevancy_indicator[i]):#there was a reward in the i'th step
            for j in (range(min(i , BACKWARDS_RELEVANCY_WINDOW))):#mark up to BACKWARDS_RELEVANCY_WINDOW
                                                                  #previous actions as relevant
                relevancy_indicator[i-j] = True

    relevancy_indicator = relevancy_indicator.astype(int)
    relevancy_indicator = np.add(np.multiply(relevancy_indicator, BOOSTING_FACTOR - 1), 1)
    return np.multiply(relevancy_indicator, processed_rewards)

#calculates rewards per step from the score per step array
def calc_reward_from_raw(score_arr):
    rewards = np.diff(score_arr)#convert raw score to points earned/lost per step
    rewards[len(rewards)-1] = PUNISHMENT_FOR_DEATH
    cumulative_discounted_rewards = np.zeros(len(rewards))
    partial_sum = 0
    #calc discounted rewards like HW4
    for i in reversed(range(0, len(rewards))):
        partial_sum = partial_sum * DISCOUNT_FACTOR + rewards[i]
        cumulative_discounted_rewards[i] = partial_sum
    #boost relevant rewards
    boosted_processed_rewards = boost( (rewards > 0) , cumulative_discounted_rewards)
    return boosted_processed_rewards

def main():
    a = np.array(range(60))
    a = (a%15 == 14)
    a= a+np.zeros(len(a))
    x = calc_reward_from_raw(a)
    print(x)

if __name__ == "__main__":
    main()