import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

REGULARIZATION_FACTOR = 10
MU = 100
TOURNAMENT_SIZE = 7
LAMBDA = TOURNAMENT_SIZE * MU
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.1
EPOCHS = 100


class Chromosome:
    def __init__(self, min_=0.0001, max_=2, chromosome_length=2):
        self.gene = np.random.uniform(min_, max_, chromosome_length)
        self.score = 0

    def evaluate(self, x, y):
        """
        Update Score Field Here
        """
        z = np.array(self.gene[0]*x+self.gene[1]*y)
        self.score = max(0, np.std(z) - REGULARIZATION_FACTOR * np.sqrt(self.gene[0]**2+self.gene[1]**2))

    def __str__(self):
        return "{}, {}".format(self.gene[0], self.gene[1])


def generate_initial_population(p_size):
    list_of_chromosomes = []
    for _ in range(p_size):
        list_of_chromosomes.append(Chromosome())
    return list_of_chromosomes


def crossover(chromosome1, chromosome2):
    p = np.random.rand()
    if p < CROSSOVER_PROBABILITY:
        ch_new1 = Chromosome()
        ch_new2 = Chromosome()
        ch_new1.gene = [chromosome1.gene[0], chromosome2.gene[1]]
        ch_new2.gene = [chromosome2.gene[0], chromosome1.gene[1]]

        return random.choice([ch_new1, ch_new2])
    else:
        return random.choice([chromosome1, chromosome2])


def mutation(chromosome):
    """
    We add a unit Gaussian distributed random value to one
    of the genes of chosen chromosome
    :param chromosome:
    :return: mutated chromosome
    """
    p = np.random.rand()
    if p < MUTATION_PROBABILITY:
        index = np.random.randint(0, 2, 1)[0]
        chromosome.gene[index] += np.random.normal()
    return chromosome


def evaluate_generation(list_of_chromosomes, x, y):
    """
    Call evaluate method for each new chromosome
    :return: probability of each chromosome for being chosen
    """
    for chromosome in list_of_chromosomes:
        chromosome.evaluate(x, y)
    return list_of_chromosomes


def select_parents(population):
    parents = []
    for _ in range(MU):
        total_score = 0
        q = np.random.choice(population, TOURNAMENT_SIZE, replace=False)
        q = sorted(q, key=lambda j: j.score, reverse=True)
        for i in range(TOURNAMENT_SIZE):
            total_score += q[i].score
        p_best = q[0].score/total_score
        probs = [p_best*(1-p_best)**j for j in range(TOURNAMENT_SIZE)]

        # slightly increasing the chance of the best chromosome in a tournament
        probs[0] += 1-sum(probs)
        winner = np.random.choice(q, 1, p=probs)[0]
        parents.append(winner)

    return parents


def create_offsprings(parents):
    children = []
    total_score_parents = 0
    prob_parents = []
    for p in parents:
        total_score_parents += p.score
        prob_parents.append(p.score)
    prob_parents = np.array(prob_parents)/total_score_parents

    for i in range(LAMBDA):
        parent1, parent2 = np.random.choice(parents, 2, replace=False, p=prob_parents)
        ch = mutation(crossover(parent1, parent2))
        children.append(ch)

    return children


def read_from_file(name):
    """
    read data points from csv
    :return: array of data
    """
    path = os.path.join(os.getcwd(), "hw2", "Dataset", name)
    df = pd.read_csv(path)
    return np.array(df["X"]), np.array(df["Y"])


def plot(x, y, chr):
    """
    Plot data points with the best vector for dimension reduction
    :return:
    """
    plt.plot(x, y, "ro", alpha=0.6)
    xl = np.linspace(0, max(x), 10)
    a, b = chr.gene
    xm, ym = np.mean(x), np.mean(y)
    m = b/(a+1e-10)
    yl = m*(xl - xm) + ym
    plt.plot(xl, yl)
    xi = (m**2/(m**2+1))*xm + (m/(m**2+1))*(y-ym) + x/(m**2+1)
    yi = m*(xi-xm) + ym
    plt.plot(xi, yi, "kx", alpha=0.6)
    plt.show()


if __name__ == '__main__':
    X, Y = read_from_file("Dataset2.csv")
    initial_population = generate_initial_population(20)
    g = evaluate_generation(initial_population, X, Y)

    for _ in range(EPOCHS):
        parents = select_parents(g)
        children = create_offsprings(parents)
        g = evaluate_generation(children, X, Y)

    sum_a = 0
    sum_b = 0
    for ch in g:
        sum_a += ch.gene[0]
        sum_b += ch.gene[1]
    sum_a, sum_b = sum_a/len(g), sum_b/len(g)
    result = Chromosome()
    result.gene = sum_a, sum_b
    result.evaluate(X, Y)

    print(result)
    print(result.score)

    plot(X, Y, result)


