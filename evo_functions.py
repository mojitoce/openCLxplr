import numpy as np
"""
The goal of this exercise is to create a genetic algorithm that
finds the global minimum of the rosenbrock function.
"""


def rosenbrock(x):
    """Convention dictates that a = 1 and b = 100 """
    f = np.power(1 - x[0], 2) + 100 * np.power(x[1] - np.power(x[0], 2), 2)
    return f



def evolution(function, iterations = 10000, initial_candiates = 30, n_parents = 20, n_children = 20, mutation = False):

    best_solution = []
    best_candidate = []
    iter = 0
    # Start with an initial population of solutions
    x = np.random.rand(2, initial_candiates) * 4 - 2
    # Calculate fitness for each of them
    f = rosenbrock(x)


    while iter < iterations:
        f = rosenbrock(x)

        best_solution.append(f.min())
        best_candidate.append(x[:, f.argmin()])

        best_parents = x[:, np.argpartition(f, n_parents)[:n_parents]]

        children = np.empty((2, n_children))
        for i in range(n_children):
            children[:, i] = np.average(best_parents[:, np.random.choice(n_children, 2)], axis = 1)

        if mutation:
            pass

        x = np.c_[best_parents, children]


        iter += 1

    return best_candidate, best_solution



    # Repeat
        # Select the best n solutions

        # Make these solutions have babies

        # Randomly mutate these solutions

        # Stop when solutions are not changing anymore


a,b = evolution(rosenbrock)


print(b[-10:], a[-10:])

print(b[-1], a[-1])