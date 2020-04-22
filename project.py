# Author: Hao Fang
# Student ID: 301301402

# TODO: Libraries - pip install wheel, pip install pandas, pip install numpy, pip install matplotlib
import time, numpy as np, random, operator, math, matplotlib.pyplot as plt, os

# Reference:
#     https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35

class City:
    def __init__(self, name, xCoord, yCoord):
        self.name = name
        self.xCoord = xCoord
        self.yCoord = yCoord
    def __repr__(self):
        return "City: " + str(self.name) + " [" + str(self.xCoord) + ", " + str(self.yCoord) + "]"
    def __str__(self):
        return "City: " + str(self.name) + " [" + str(self.xCoord) + ", " + str(self.yCoord) + "]"

class Individual:
    def __init__(self, individual, score):
        self.individual = individual
        self.score = score
    def __getitem__(self, item):
        return self.score
    def __repr__(self):
        return "Individual Score: " + str(1 / self.score) + "\n"
    def __str__(self):
        return "Individual Score: " + str(1 / self.score) + "\n"

"""
This will rank the individuals in a population
The score is calculated in reverse order of the entire distance
The bigger the score the shorter the distance the the smaller the index
"""
def rankPopulation(population):
    # rankList = []
    # for individual in population:
    #     rankList.append(Individual(individual, 1 / calIndividualScore(individual)))
    return sorted(population, key=operator.itemgetter(0), reverse=True)

"""
The best individuals will have a higher chance of being chosen
This is one of the modified version which includes elitism.
(It will move a number of elitists to the selection array before
we run the algorithm)

Those selected individuals will be the parent for next generation
"""
def getMatingPool(populationList, elitismSize):
    selectionList = []
    scoreList = []
    # generate a list of reverse of scores
    for num in range(0, len(populationList)):
        scoreList.append(populationList[num].score)
    # transfer the list in to numpy list
    scoreList = np.array(scoreList)
    cumSumList = np.array(scoreList).cumsum()
    totalSum = scoreList.sum()

    for num in range(0, elitismSize):
        selectionList.append(populationList[num].individual)
    for num in range(0, len(populationList) - elitismSize):
        randomNum = 100 * random.random()
        for i in range(0, len(populationList)):
            if randomNum <= 100 * cumSumList[i] / totalSum:
                selectionList.append(populationList[i].individual)
                break
    return selectionList

"""
Randomly select a subset of the first parent string 
and then fill the reminder of the route with the genes
from the second parent
"""
def generateChildren(parent1, parent2):
    childP1 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def generateNewPopulation(matingPool, eliteSize):
    children = []
    length = len(matingPool) - eliteSize
    pool = random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])

    for i in range(0, length):
        child = generateChildren(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPopulation = []
    for index in range(0, len(population)):
        mutatedIndividual = mutate(population[index], mutationRate)
        mutatedPopulation.append(Individual(mutatedIndividual, 1 / calIndividualScore(mutatedIndividual)))
    return mutatedPopulation

def generateNextPopulation(currentGeneration, eliteSize, mutationRate):
    rankedPopulation = rankPopulation(currentGeneration)
    matingPool = getMatingPool(rankedPopulation, eliteSize)
    children = generateNewPopulation(matingPool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(initialIndividual, populationSize, eliteSize, mutationRate, numGenerations):
    currentPopulation = []
    for i in range(0, populationSize):
        random.shuffle(initialIndividual)
        tempList = list(initialIndividual)
        currentPopulation.append(Individual(tempList, 1 / calIndividualScore(tempList)))

    for num in range(0, numGenerations):
        currentPopulation = generateNextPopulation(currentPopulation, eliteSize, mutationRate)

    print(currentPopulation)
    maxScore = currentPopulation[0].score
    bestIndex = 0
    for num in range(0, len(currentPopulation)):
        if maxScore < currentPopulation[num].score:
            bestIndex = num
            maxScore = currentPopulation[num].score
    write("best-solution1000_fanghaof.txt", currentPopulation[bestIndex].individual, 1 / currentPopulation[bestIndex].score)

def geneticAlgorithmWithGraph(initialIndividual, populationSize, eliteSize, mutationRate, numGenerations):
    currentPopulation = []
    progress = []
    for i in range(0, populationSize):
        random.shuffle(initialIndividual)
        tempList = list(initialIndividual)
        currentPopulation.append(Individual(tempList, 1 / calIndividualScore(tempList)))
    progress.append(1 / rankPopulation(currentPopulation)[0].score)

    for num in range(0, numGenerations):
        currentPopulation = generateNextPopulation(currentPopulation, eliteSize, mutationRate)
        temp = rankPopulation(currentPopulation)[0]
        progress.append(1 / temp.score)
        print(1 / temp.score)
        temp = temp.individual
        for cityIndex in range(len(temp) - 1):
            print(temp[cityIndex].name + " ", end="")
        print(temp[len(temp) - 1].name)

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    rankedPopulation = rankPopulation(currentPopulation)
    info = rankedPopulation[0]
    updateRecord("best-solution1000_fanghaof.txt", info.individual, 1 / info.score)

def pmx(s, t):
    n = len(s)

    # choose crossover point at random
    c = random.randrange(1, n - 1)  # c is index of last element of left part

    # first offspring
    first = s[:]
    i = 0
    while i <= c:
        j = first.index(t[i])
        first[i], first[j] = first[j], first[i]
        i += 1

    # second offspring
    second = t[:]
    i = 0
    while i <= c:
        j = second.index(s[i])
        second[i], second[j] = second[j], second[i]
        i += 1

    return first, second

def do_rand_swap(lst):
    n = len(lst)
    i, j = random.randrange(n), random.randrange(n)
    lst[i], lst[j] = lst[j], lst[i]  # swap lst[i] and lst[j]
    return lst

def crossover(mum, dad):
    """Implements ordered crossover"""

    size = len(mum)

    # Choose random start/end position for crossover
    alice, bob = [-1] * size, [-1] * size
    start, end = sorted([random.randrange(size) for _ in range(2)])

    # Replicate mum's sequence for alice, dad's sequence for bob
    alice_inherited = []
    bob_inherited = []
    for i in range(start, end + 1):
        alice[i] = mum[i]
        bob[i] = dad[i]
        alice_inherited.append(mum[i])
        bob_inherited.append(dad[i])

    print(alice, bob)
    #Fill the remaining position with the other parents' entries
    current_dad_position, current_mum_position = 0, 0

    fixed_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        if i in fixed_pos:
            i += 1
            continue

        test_alice = alice[i]
        if test_alice==-1: #to be filled
            dad_trait = dad[current_dad_position]
            while dad_trait in alice_inherited:
                current_dad_position += 1
                dad_trait = dad[current_dad_position]
            alice[i] = dad_trait
            alice_inherited.append(dad_trait)

        #repeat block for bob and mom
        i +=1

    return alice, bob

def generateNextPopulationOptimized(population, populationSize):
    nextGeneration = []
    for i in range(0, int(populationSize * 0.2)):
        temp = Individual(population[i].individual, population[i].score)
        nextGeneration.append(temp)

    for i in range(0, int(populationSize * 0.2 / 2)):
        a = list(random.choice(nextGeneration).individual)
        b = list(random.choice(nextGeneration).individual)
        childA, childB = pmx(a, b)
        nextGeneration.append(Individual(childA, 1 / calIndividualScore(childA)))
        nextGeneration.append(Individual(childB, 1 / calIndividualScore(childB)))

    for i in range(0, int(populationSize * 0.5)):
        a = list(random.choice(nextGeneration).individual)
        b = list(random.choice(nextGeneration).individual)
        child = generateChildren(a, b)
        nextGeneration.append(Individual(child, 1 / calIndividualScore(child)))

    while len(nextGeneration) < populationSize:
        a = list(nextGeneration[0].individual)
        for i in range(0, 10):
            a = do_rand_swap(a)
        nextGeneration.append(Individual(a, 1 / calIndividualScore(a)))

    return nextGeneration

def geneticAlgorithmWithPMXCustomized(initialIndividual, populationSize, numGenerations):
    currentPopulation = []
    progress = []
    for i in range(0, int(populationSize / 5)):
        currentPopulation.append(Individual(initialIndividual, 1 / calIndividualScore(initialIndividual)))
    progress.append(1 / currentPopulation[0].score)

    while len(currentPopulation) < populationSize:
        random.shuffle(initialIndividual)
        tempList = list(initialIndividual)
        currentPopulation.append(Individual(tempList, 1 / calIndividualScore(tempList)))

    for num in range(0, numGenerations):
        print("iteration: ", num)
        currentPopulation = generateNextPopulationOptimized(currentPopulation, populationSize)
        currentPopulation = rankPopulation(currentPopulation)
        temp = currentPopulation[0]
        progress.append(1 / temp.score)
        print(1 / temp.score)
        temp = temp.individual
        for cityIndex in range(len(temp) - 1):
            print(temp[cityIndex].name + " ", end="")
        print(temp[len(temp) - 1].name)

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
    rankedPopulation = rankPopulation(currentPopulation)
    info = rankedPopulation[0]
    updateRecord("best-solution1000_fanghaof.txt", info.individual, 1 / info.score)

def updateRecord(fileName, bestIndividual, bestScore):
    try:
        userFile = open(fileName, "r+")
    except FileNotFoundError:
        print("Error: %s does not exits", fileName)
        return
    except IOError:
        print("Error: IOError detected", fileName)
        return

    tokens = userFile.readline().replace("\n", "").split(" ")
    if (os.stat(fileName).st_size != 0) and (float(tokens[1]) <= float(bestScore)):
        return
    else:
        userFile.truncate(0)
        userFile.seek(0, os.SEEK_SET)
        userFile.write("Distance: " + str(bestScore) + "\n")
        for cityIndex in range(len(bestIndividual) - 1):
            userFile.write(bestIndividual[cityIndex].name + " ")
        userFile.writelines(bestIndividual[len(bestIndividual) - 1].name)
        userFile.close()

def printGenerationDetail(myGeneration):
    for num in range(0, len(myGeneration)):
        print("Individual Score ", num, ": ", calIndividualScore(myGeneration[num]))

def load(fileName):
    cityList = []
    try:
        userFile = open(fileName, "r")
    except FileNotFoundError:
        print("Error: %s does not exits", fileName)
        return
    except IOError:
        print("Error: IOError detected", fileName)
        return

    lines = userFile.read().splitlines()
    for line in lines:
        tokens = line.split(" ")
        cityList.append(City(tokens[0], int(tokens[1]), int(tokens[2])))
    userFile.close()
    return cityList

def write(fileName, info, score):
    try:
        userFile = open(fileName, "a+")
    except IOError:
        print("Error: IOError detected", fileName)
        return

    userFile.writelines("Total distance: " + str(score) + "\n")
    for cityIndex in range(len(info) - 1):
        userFile.write(info[cityIndex].name + " ")
    userFile.writelines(info[len(info) - 1].name)
    userFile.writelines("\n")
    userFile.close()

def calIndividualScore(cityList):
    score = 0
    for cityIndex in range(len(cityList) - 1):
        score += math.sqrt(
            math.pow(cityList[cityIndex].xCoord - cityList[cityIndex + 1].xCoord, 2)
            + math.pow(cityList[cityIndex].yCoord - cityList[cityIndex + 1].yCoord, 2))
    # add the distance between last city to the first city
    score += math.sqrt(
        math.pow(cityList[len(cityList) - 1].xCoord - cityList[0].xCoord, 2)
        + math.pow(cityList[len(cityList) - 1].yCoord - cityList[0].yCoord, 2))
    return score

if __name__ == '__main__':
    cities = load("cities1000.txt")

    # cities = []
    # userFile = open("best-solution1000_fanghaof.txt", "r")
    # lines = userFile.readline()
    # userFile.close()
    # tokens = lines.split(" ")
    # for city in tokens:
    #     a = int(city)
    #     for i in original_cities:
    #         if a == int(i.name):
    #             cities.append(City(i.name, i.xCoord, i.yCoord))

    print("Initial Score: ", calIndividualScore(cities))
    start = time.time()
    geneticAlgorithmWithPMXCustomized(cities, 500, 100000)
    # geneticAlgorithmWithGraph(cities, 500, 200, 0.00, 100000)
    end = time.time()
    print("Time Elapsed: ", end - start)
