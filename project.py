# Author: Hao Fang
# Student ID: 301301402

# TODO: Libraries - pip install wheel, pip install pandas, pip install numpy, pip install matplotlib
import numpy as np, random, operator, math, pandas as pd, matplotlib.pyplot as plt, os

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

def geneticAlgorithmWithPMX(initialIndividual, populationSize, eliteSize, mutationRate, numGenerations):
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

cities = load("cities1000.txt")
# cities = load("testCities10.txt")
print("Initial Score: ", calIndividualScore(cities))
# geneticAlgorithm(cities, 100, 10, 0.01, 2000)
geneticAlgorithmWithGraph(cities, 500, 100, 0.05, 2000)
# test = Fitness(1, 1)
# test1 = Fitness(2, 3)
# test2 = Fitness(3, 2)
# List = [test, test1, test2]
# test12 = sorted(List, key=operator.itemgetter(1), reverse=False)
# for element in test12:
#     print(element.individual)
# testList = [1, 2, 3, 4, 5]
# test = np.array(testList)