import numpy as np
#hyper parameters for reward function
DISCOUNT_FACTOR = 0.99
#BACKWARDS_RELEVANCY_WINDOW = 10
#BOOSTING_FACTOR = 3.0
PUNISHMENT_FOR_DEATH = -100
NO_REWARD_PENALTY = -1
SMALL_GAIN_REWARD = 2
MEDIUM_GAIN_REWARD = 5
LARGE_GAIN_REWARD = 10

'''
#boosts by BOOSTING_FACTOR all the BACKWARDS_RELEVANCY_WINDOW rewards before an
#actual point was earned in the game , and rewards at step when accelaration took place
def boost(reward_indicator, accel_penalty_indicator, processed_rewards):
    for i in (range(len(reward_indicator))):
        if(reward_indicator[i]):#there was a reward in the i'th step
            for j in (range(min(i , BACKWARDS_RELEVANCY_WINDOW))):#mark up to BACKWARDS_RELEVANCY_WINDOW
                                                                  #previous actions as relevant
                reward_indicator[i - j] = True

    boosting_indicator = (np.logical_or( reward_indicator , accel_penalty_indicator)).astype(np.int)
    boosting_indicator = np.add(np.multiply(boosting_indicator, BOOSTING_FACTOR - 1), 1)
    return np.multiply(boosting_indicator , processed_rewards)
'''
#calculates rewards per step from the score per step array
#TODO: need to take care when the array is at size 2
def calc_reward_from_raw(score_arr , is_dead):
    if (len(score_arr) == 1):
        return np.array([0]) # worst case TODO : i think should never happen im model
    rewards = []
    scores_diff = np.diff(score_arr)    #convert raw score to points earned/lost per step
    for k in scores_diff:
        if(k<=0):
            rewards.append(NO_REWARD_PENALTY)
        elif(k>0 and k<=5):
            rewards.append(SMALL_GAIN_REWARD)
        elif(k>5 and k<=15):
            rewards.append(MEDIUM_GAIN_REWARD)
        else:
            rewards.append(LARGE_GAIN_REWARD)


    if (is_dead):
        rewards[len(rewards)-1] = PUNISHMENT_FOR_DEATH      #TODO: I am not sure this is true, we shouldn't update only in death

    #compute discounted rewards
    discounted_formula = np.frompyfunc(lambda x, y: DISCOUNT_FACTOR * x + y, 2, 1)
    cumulative_discounted_rewards= np.flip(
                discounted_formula.accumulate(np.flip(rewards, 0), dtype=np.object).astype(np.double),0)

    #boost relevant rewards
    #boosted_processed_rewards = boost( (rewards > 0) ,(rewards < NO_REWARD_PENALTY)
    #
    #                                    , cumulative_discounted_rewards)
    #normalize
    normalized_rewards = np.divide(cumulative_discounted_rewards, np.std(cumulative_discounted_rewards))
    normalized_rewards = np.add (normalized_rewards , -1)

    return normalized_rewards

def main():
    a = np.array(range(60))
    a = np.diff(np.cumsum(np.add((a%15==14).astype(np.int) , np.multiply((a%30==28).astype(np.int) , -1))))
    print (a)

    print(calc_reward_from_raw(a))

if __name__ == "__main__":
    main()