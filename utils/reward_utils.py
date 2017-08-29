import numpy as np
#hyper parameters for reward function
DISCOUNT_FACTOR = 0.99
#BACKWARDS_RELEVANCY_WINDOW = 10
#BOOSTING_FACTOR = 3.0
PUNISHMENT_FOR_DEATH = -10
NO_REWARD_PENALTY = -0.05
#SMALL_GAIN_REWARD = 2
#MEDIUM_GAIN_REWARD = 5
#LARGE_GAIN_REWARD = 10

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
def discount(rewards):
    # compute discounted rewards
    cumulative_discounted_rewards = np.zeros(len(rewards))
    partial_sum = 0.0
    for i in reversed(range(0, len(rewards))):
        partial_sum = partial_sum * DISCOUNT_FACTOR + rewards[i]
        cumulative_discounted_rewards[i] = partial_sum
    return cumulative_discounted_rewards

#calculates rewards per step from the score per step array
def calc_reward_from_raw(score_arr , is_dead):
    rewards = np.diff(score_arr)    #convert raw score to points earned/lost per step
    rewards = np.add( rewards , np.multiply( (rewards == 0) ,NO_REWARD_PENALTY))

    if is_dead:
        rewards[-1] = PUNISHMENT_FOR_DEATH

    cumulative_discounted_rewards = discount(rewards)
    #boost relevant rewards
    #boosted_processed_rewards = boost( (rewards > 0) ,(rewards < NO_REWARD_PENALTY)
    #                                    , cumulative_discounted_rewards)
    #normalize
    cumulative_discounted_rewards /= np.std(cumulative_discounted_rewards)
    cumulative_discounted_rewards -= np.mean(cumulative_discounted_rewards)

    return cumulative_discounted_rewards

def main():
    a = np.array(range(60))
    a = np.diff(np.cumsum(np.add((a%15==14).astype(np.int) , np.multiply((a%30==28).astype(np.int) , -1))))
    a*= 1
    print (a)

    print(calc_reward_from_raw(a, True))

if __name__ == "__main__":
    main()

# discounting using numpy
'''
    # compute discounted rewards
    discounted_formula = np.frompyfunc(lambda x, y: DISCOUNT_FACTOR * x + y, 2, 1)
    cumulative_discounted_rewards = np.flip(
        discounted_formula.accumulate(np.flip(rewards, 0), dtype=np.object).astype(np.double), 0)
'''

'''
    for k in scores_diff:
        if(k<=0):
            rewards.append(NO_REWARD_PENALTY)
        elif(k>0 and k<=5):
            rewards.append(SMALL_GAIN_REWARD)
        elif(k>5 and k<=15):
            rewards.append(MEDIUM_GAIN_REWARD)
        else:
            rewards.append(LARGE_GAIN_REWARD)
'''