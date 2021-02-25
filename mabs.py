import numpy as np
import random
from preprocessing import prepare_data, assign_to_clusters
from scipy.stats import beta


class MAB:
    def __init__(self):
        x_train, x_test, y_train, y_test = prepare_data()
        self.arms_data = assign_to_clusters(x_train, y_train)
        self.arms_names = sorted(self.arms_data.keys())
        self.n_arms = len(self.arms_names)
        self.current_arm = 0
        self.x_test = x_test
        self.y_test = y_test

    def get_batch(self, batch_size):
        """
        Generates a batch_size of data from the selected arm (to make the process quicker)
        :param batch_size: int
        :return: np.array (batch_size, 28,28, 1)
        :return: np.array (batch_size,)
        """
        arm = self.current_arm
        all_idxs = np.arange(self.arms_data[arm]['x'].shape[0])
        batch_idxs = np.random.choice(all_idxs, size=batch_size, replace=False)
        x_ = self.arms_data[arm]['x'][batch_idxs]
        y_ = self.arms_data[arm]['y'][batch_idxs]
        return x_, y_

    def select_arm(self):
        """
        Selects the current arm (cluster) according to the algorithm
        :return:
        """
        pass

    def update(self, reward):
        """
        Updates the weights of the algorithm
        :param reward: float, the improvement in test_acc from last iteration (clipped between 0-1)
        :return:
        """
        pass


class MabRandomArm(MAB):
    def __init__(self):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.rewards = [[] for i in range(self.n_arms)]


    def select_arm(self):
        self.current_arm = np.random.choice(self.arms_names)
        self.counts[self.current_arm] += 1
      
    def update(self, reward):
        arm = self.current_arm
        self.rewards[arm].append(reward)


class MabEpsilonGreedy(MAB):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def select_arm(self):
        # If prob is not in epsilon, do exploitation of best arm so far
        if random.random() > self.epsilon:
            self.current_arm = np.argmax(self.values)
        # If prob falls in epsilon range, do exploration
        else:
            self.current_arm = random.randrange(len(self.values))

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value


class MabUcb1(MAB):
    def __init__(self):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.rewards = [[] for i in range(self.n_arms)]

    def select_arm(self):
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.rewards[arm].append(reward)

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value


class MabSWUcb1(MAB):
    def __init__(self, n):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.n = n
        self.rewards = [[] for i in range(self.n_arms)]

    def select_arm(self):
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] = min(self.counts[arm]+1,self.n)
        n = self.counts[arm]
        self.rewards[arm].append(reward+random.random()/1000)

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = np.mean(self.rewards[arm]) if n<self.n else np.mean(self.rewards[arm][-self.n:])
        self.values[arm] = new_value

class ExpMabUcb1(MAB):
    def __init__(self):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def select_arm(self):
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] += 1
        n = self.counts[arm]

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = 0.9 * value + 0.1 * reward
        self.values[arm] = new_value

class MabThompsonSampling(MAB):
    def __init__(self):
        super().__init__()

        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

        # Uniform distribution of prior beta (A=1,B=1)
        self.a = np.ones(self.n_arms)
        self.b = np.ones(self.n_arms)

    def select_arm(self):

        # Pair up all Beta Distribution params of a and b for each arm
        beta_params = zip(self.a, self.b)

        # Perform random draw for all arms based on their params (a,b)
        all_draws = [beta.rvs(i[0], i[1], size=1) for i in beta_params]
        all_draws = np.array(all_draws)

        # return index of arm with the highest draw
        self.current_arm = np.argmax(all_draws)

    def update(self, reward):

        # Make reward binary for Beta distribution
        # Using a Gaussian Thompson Sampling requires us to know the variance or mean. And we cannot know this
        reward = 0.0 if reward <= 0 else 1.0
        arm = self.current_arm

        # update counts for chosen arm
        self.counts[arm] = self.counts[arm] + 1
        n = self.counts[arm]

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value

        # Update a and b
        # a is based on total counts of rewards of arm
        self.a[arm] += reward

        # b is based on total counts of failed rewards on arm
        self.b[arm] += (1 - reward)


class MabExp3(MAB):
    def __init__(self, gamma):
        super().__init__()
        self.weights = np.ones(self.n_arms)
        self.gamma = gamma
        self.dist = self.weights/self.n_arms
        self.counts = np.zeros(self.n_arms)

    def select_arm(self):

        # Get probability distribution
        self.dist = (1 - self.gamma) * (self.weights / self.weights.sum()) + (self.gamma / self.n_arms)

        # Draw from distribution
        self.current_arm = np.random.choice(np.arange(self.n_arms), p=self.dist)

    def update(self, reward):

        arm = self.current_arm
        self.counts[arm] += 1

        # Unbiased estimate for reward
        r_hat = reward / self.dist[arm]

        # Update weight of selected arm (all other arms remain the same)
        self.weights[arm] *= np.exp(r_hat * self.gamma / self.n_arms)


class MabExp3Ix(MAB):
    def __init__(self, gamma, eta):
        super().__init__()
        self.weights = np.ones(self.n_arms)
        self.gamma = gamma
        self.eta = eta
        self.dist = self.weights / self.n_arms

    def select_arm(self):
        # Get probability distribution
        self.dist = self.weights / self.weights.sum()

        # Draw from distribution
        self.current_arm = np.random.choice(np.arange(self.n_arms), p=self.dist)

    def update(self, reward):
        arm = self.current_arm

        # Unbiased estimate for reward
        r_hat = reward / (self.dist[arm] + self.gamma)

        # Update weight of selected arm (all other arms remain the same)
        self.weights[arm] *= np.exp(r_hat * self.eta)


class MabFtl(MAB):
    def __init__(self, eta):
        super().__init__()
        self.values = np.zeros(self.n_arms)
        self.eta = eta

    def select_arm(self):
        noise = np.random.exponential(1/self.eta, size=self.n_arms)
        self.current_arm = np.argmax(self.values + noise)

    def update(self, reward):
        arm = self.current_arm

        # Update values of selected arm (all other arms remain the same)
        self.values[arm] += reward
        
 class MabEpsUcb1(MAB):
    def __init__(self, eps):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.rewards = [[] for i in range(self.n_arms)]
        self.eps = eps
        self.last_play = np.zeros(self.n_arms)
        self.round = 0

    def select_arm(self):
      if random.random()>self.eps:
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)
      else:
        self.current_arm = np.random.choice(np.arange(self.n_arms), p=(self.round - self.last_play)/np.sum((self.round - self.last_play)))

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.rewards[arm].append(reward)
        self.round += 1
        self.last_play[arm] = self.round

        # Update mean reward for chosen arm
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        
class MabCDUcb1(MAB):
    def __init__(self, n):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.n = n
        self.rewards = [[] for i in range(self.n_arms)]

    def select_arm(self):
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] = self.counts[arm]+1
        n = self.counts[arm]
        self.rewards[arm].append(reward)

        # Update mean reward for chosen arm
        value = self.values[arm]
        cond = abs(np.mean(self.rewards[arm])-np.mean(self.rewards[arm][-self.n:]))
        if cond<0.1:
          new_value = np.mean(self.rewards[arm])  
        else:
          print('restart:',arm,np.mean(self.rewards[arm]), np.mean(self.rewards[arm][-self.n:]), np.mean(self.rewards[arm][-10:]), n)
          new_value = np.mean(self.rewards[arm][-self.n:])
          self.counts[arm] = self.n
          self.rewards[arm] = self.rewards[arm][-self.n:]
        self.values[arm] = new_value
        
        
 class MabEpsCDUcb1(MAB):
    def __init__(self, eps, n):
        super().__init__()
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.rewards = [[] for i in range(self.n_arms)]
        self.eps = eps
        self.last_play = np.zeros(self.n_arms)
        self.round = 0
        self.n = n

    def select_arm(self):
      if random.random()>self.eps or self.round<5:
        ucb_values = np.zeros(self.n_arms)
        total_counts = self.counts.sum()

        for arm in range(self.n_arms):
            value = self.values[arm] + np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm] + 1))
            ucb_values[arm] = value

        self.current_arm = np.argmax(ucb_values)
      else:
        self.current_arm = np.random.choice(np.arange(self.n_arms), p=(self.round - self.last_play)/np.sum((self.round - self.last_play)))

    def update(self, reward):
        arm = self.current_arm
        # update counts for chosen arm
        self.counts[arm] += 1
        n = self.counts[arm]
        self.rewards[arm].append(reward)
        self.round += 1
        self.last_play[arm] = self.round

        # Update mean reward for chosen arm
        value = self.values[arm]
        cond = abs(np.mean(self.rewards[arm])-np.mean(self.rewards[arm][-self.n:]))
        if cond<0.1:
          new_value = np.mean(self.rewards[arm])  
        else:
          print('restart:',arm,np.mean(self.rewards[arm]), np.mean(self.rewards[arm][-self.n:]), np.mean(self.rewards[arm][-10:]), n)
          new_value = np.mean(self.rewards[arm][-self.n:])
          self.counts[arm] = self.n
          self.rewards[arm] = self.rewards[arm][-self.n:]
        self.values[arm] = new_value
