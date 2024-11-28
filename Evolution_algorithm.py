import random
import matplotlib.pyplot as plt

def genetic_algorithm():
    Kcap = 50
    POP_SIZE = 30
    GEN_MAX = 5000000
    NUM_ITEMS = 15

    ITEMS = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(NUM_ITEMS)]

    generation = 1
    population = Initialization(POP_SIZE, NUM_ITEMS)

    # List to store total fitness values over generations
    fitness_over_generations = []

    for x in range(GEN_MAX):
        population = sorted(population, key=lambda ind: fitness(ind, ITEMS, Kcap), reverse=True)
        total_fitness = sum(fitness(ind, ITEMS, Kcap) for ind in population)
        fitness_over_generations.append(total_fitness)
        print(f'Downloading  {generation}, Whatsapp chat: {total_fitness}')
        
        population = Evolution(population)
        generation += 1

    # Final sorting to determine the best individual
    population = sorted(population, key=lambda k: fitness(k, ITEMS, Kcap), reverse=True)
    print('Best Individual:', population[0])

    # Plotting the fitness over generations
    plt.plot(range(1, GEN_MAX + 1), fitness_over_generations)
    plt.xlabel('Generation')
    plt.ylabel('Total Fitness')
    plt.title('Total Fitness over Generations')
    plt.show()

# Step 1: Initialization phase
def Initialization(POP_SIZE, NUM_ITEMS):
    population = []
    for _ in range(POP_SIZE):
        individual = [random.randint(0, 1) for _ in range(NUM_ITEMS)]
        population.append(individual)
    return population

# Step 2: Evaluation and fitness phase
def fitness(individual, ITEMS, Kcap):
    total_value = 0 
    total_weight = 0

    for idx, bit in enumerate(individual):
        if bit == 1:
            total_value += ITEMS[idx][0]
            total_weight += ITEMS[idx][1]

    if total_weight > Kcap:
        return 0
    else:
        return total_value

# Step 4: Evolution phase
def Evolution(population):
    parent_percent = 0.2
    mutation_rate = 0.08
    parent_lottery = 0.05

    # Selection
    parent_length = int(parent_percent * len(population))
    parents = population[:parent_length]

    # Crossover
    children = []
    desired_length = len(population) - len(parents)
    while len(children) < desired_length:
        male = random.choice(parents)
        female = random.choice(parents)
        half = len(male) // 2
        child = male[:half] + female[half:]
        children.append(child)

    # Mutation
    for child in children:
        for i in range(len(child)):
            if mutation_rate > random.random():
                child[i] = 1 - child[i]

    parents.extend(children)
    return parents

if __name__ == "__genetic_algorithm__":
    genetic_algorithm()

genetic_algorithm()