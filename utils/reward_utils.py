import numpy as np

#hyper parameters for reward function
DISCOUNT_FACTOR = 0.99
PUNISHMENT_FOR_DEATH = -10
NO_REWARD_PENALTY = -0.05

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