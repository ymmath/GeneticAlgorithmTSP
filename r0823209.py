import random

import Reporter
import numpy as np
import time


class r0823209:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.populationSize = 1500  # isn't used anymore, it is determined as a function of the length of the distance matrix
        self.elitismRate = 0.05
        self.tournamentSize = 3
        self.mutationRate = 0.025
        self.nnPercent = 0.1  # Proportion of the initial population that will be created heuristically (ignored if it is larger than the number of cities)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        start = time.time()
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        size = len(distanceMatrix)
        self.populationSize = int(1790.408 - 2.58678 * size + 0.001409554 * (size ** 2))

        nnCandidates = int(min(distanceMatrix.shape[0], self.nnPercent * self.populationSize))
        # print("NN Candidates: " + str(nnCandidates))
        population = self.createInitialPopulation(distanceMatrix, nnCandidates)
        previousWinner = population[0]
        ct = 0
        iterNum = 0
        iterChange = []
        yourConvergenceTestsHere = True
        initialmuteRate = self.mutationRate
        initialK = self.tournamentSize

        while (yourConvergenceTestsHere):

            self.tournamentSize = min(int(initialK + 0.5 * iterNum),
                                      int(0.025 * self.populationSize))  # adapting the K (in tournament selection) as a function of the iteration number
            iterNum += 1
            self.mutationRate = initialmuteRate * (
                        1 + np.log10(iterNum + 1))  # adapting the mutation rate as a function of the iteration number

            winners = self.takeBestOnes(population)  # elitism
            matingPool = [self.selection(population) for _ in range(
                self.populationSize)]  # K tournament selection (adjust it to only create range((1 - self.elitismRate)*self.populationSize) tours to reduce the computation time)
            matingPoolTours = [a_tuple[0] for a_tuple in matingPool]

            crossovers = self.crossover(matingPoolTours)  # Current crossover: OX
            mutations = self.mutate(crossovers, distanceMatrix)  # Current mutation: Inversion
            leftovers = self.eliminate(mutations, distanceMatrix)  # Current elimination: Elitism

            # add local search here for to 10% of the Elitists
            newPopulation = []
            # for i in range(int(0.04*len(winners))):
            # 	routeLocal, distLocal = self.two_opt(winners[i][0], winners[i][1], distanceMatrix, 0.9)
            # 	newPopulation.append((routeLocal, distLocal))
            for i in range(3):
                if i == 0:
                    anArray = winners.pop()  # pop the first one
                    routeLocal, distLocal = self.two_opt(anArray[0], anArray[1], distanceMatrix,
                                                         0.01)  # do the local search
                    newPopulation.append((routeLocal, distLocal))  # append it
                else:
                    lucky = random.sample(range(len(winners)), 1)  # sample a random index
                    anArray = winners.pop(lucky[0])  # pop that one
                    routeLocal, distLocal = self.two_opt(anArray[0], anArray[1], distanceMatrix,
                                                         0.01)  # do the local search
                    newPopulation.append((routeLocal, distLocal))  # append it

            newPopulation = newPopulation + leftovers + winners  # [int(0.04*len(winners)):]
            population = sorted(newPopulation, key=lambda x: x[1])

            newWinner = population[0]
            epsilon = previousWinner[1] - newWinner[1]
            iterChange.append(epsilon)
            if epsilon < 0.0001:
                ct = ct + 1
            else:
                ct = 0
            previousWinner = newWinner
            if ct == 12:
                bestSolution, bestObjective = self.two_opt_full(newWinner[0], newWinner[1], distanceMatrix)  # final opt
                yourConvergenceTestsHere = False
                end = time.time()
                print("time:")
                timing = end - start
                print(timing)

            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            if (yourConvergenceTestsHere == False):
                bestSolution = [x - 1 for x in bestSolution]
                meanObjective = self.calculateMean(population)
                # print(str(iterNum) + ": Best solution:", bestSolution, '\n', bestObjective, '\nMean:', meanObjective)
                timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution))
            else:
                bestObjective = newWinner[1]
                bestSolution = [x - 1 for x in newWinner[0]]
                # bestSolution.append(bestSolution[0])
                meanObjective = self.calculateMean(population)
                # print(str(iterNum) + ": Best solution:", newWinner[0], '\n', newWinner[1], '\nMean:', meanObjective)
                timeLeft = self.reporter.report(meanObjective, bestObjective, np.array(bestSolution))

            if timeLeft < 0:
                end = time.time()
                print("time:")
                timing = end - start
                print(timing)
                # print("Best solution:", newWinner[1])
                break

        # Your code here.
        # print("Mean improvement per iteration:", np.mean(iterChange))
        return bestObjective

    def createInitialPopulation(self, distanceMatrix, nnCandidates):
        size = len(distanceMatrix)
        population = []

        # create candidates with the NN heuristic
        if nnCandidates == size:
            for i in range(1, nnCandidates + 1):
                tour = self.NN(distanceMatrix, start=i).tolist()
                distance = self.length(distanceMatrix, tour)
                population.append((tour, distance))
        else:
            for i in range(1, nnCandidates + 1):
                tour = self.NN(distanceMatrix,
                               start=np.random.randint(1, size + 1)).tolist()  # start at any random city
                distance = self.length(distanceMatrix, tour)
                population.append((tour, distance))

        # create the remaining canditates randomly
        for i in range(0, self.populationSize - nnCandidates):
            tour = self.createTour(size)
            distance = self.length(distanceMatrix, tour)
            population.append((tour, distance))

        # sort and return the population
        population = sorted(population, key=lambda x: x[1])
        return population

    def createTour(self, number):
        numbers = list(range(1, number + 1))
        new = random.sample(numbers, len(numbers))
        return new

    def two_opt(self, route, distance, distanceMatrix, improvement_threshold=0.1):
        improvement_factor = 1
        best_distance = distance
        while improvement_factor > improvement_threshold:
            distance_to_beat = best_distance
            for swap_first in range(1, len(route) - 1):
                for swap_last in range(swap_first + 1, swap_first + 2):
                    new_route = self.two_opt_swap(route.copy(), swap_first,
                                                  swap_last)
                    new_distance = self.length(distanceMatrix,
                                               new_route)
                    if new_distance < best_distance:
                        route = new_route.copy()
                        best_distance = new_distance
            improvement_factor = 1 - best_distance / distance_to_beat
        return list(route), best_distance

    def two_opt_full(self, route, distance, distanceMatrix):
        # improvement_factor = 1
        best_distance = distance
        # while improvement_factor > improvement_threshold:
        # distance_to_beat = best_distance
        for swap_first in range(1, len(route) - 1):
            for swap_last in range(swap_first + 1, len(route)):
                new_route = self.two_opt_swap(route.copy(), swap_first,
                                              swap_last)
                new_distance = self.length(distanceMatrix,
                                           new_route)
                if new_distance < best_distance:
                    route = new_route.copy()
                    best_distance = new_distance
        # improvement_factor = 1 - best_distance/distance_to_beat
        return list(route), best_distance

    # def two_opt_swap(self, r, i, k):
    # 	return np.concatenate((r[0:i], r[k:-len(r)+i-1:-1], r[k+1:len(r)])).astype(int)
    def two_opt_swap(self, toFlip, i, k):
        toFlip[i:k + 1] = toFlip[i:k + 1][::-1]
        return toFlip

    def NN(self, A, start):
        start = start - 1  # change to the representation required by the function
        path = [start]
        # cost = 0
        N = A.shape[0]
        mask = np.ones(N, dtype=bool)  # boolean values indicating which
        # locations have not been visited
        mask[start] = False

        for i in range(N - 1):
            last = path[-1]
            next_ind = np.argmin(A[last][mask])  # find minimum of remaining locations
            next_loc = np.arange(N)[mask][next_ind]  # convert to original location
            path.append(next_loc)
            mask[next_loc] = False
            # cost += A[last, next_loc]

        # cost += A[next_loc, start] # return to the starting location
        path = path + np.ones(N, dtype=int)  # change back to the representation for the user
        return path

    def length(self, distanceMatrix, tour):
        totalDistance = 0
        for i in range(0, len(tour)):
            fromCity = tour[i]
            toCity = None
            if i + 1 < len(tour):
                toCity = tour[i + 1]
            else:
                toCity = tour[0]
            totalDistance += distanceMatrix[fromCity - 1][toCity - 1]
        return totalDistance

    # alternate function for calculating distance
    # def path_distance(self, c, r):
    # 	r = (r - np.ones(len(r))).astype(int)
    # 	return np.sum([c[r[p]][r[p+1]] for p in range(len(r)-1)])+c[r[int(len(r)-1)]][r[0]]

    def takeBestOnes(self, population):
        count = self.elitismRate * self.populationSize
        bestOnes = population[:int(count)]
        return bestOnes

    def selection(self, population):
        parents = random.choices(population, k=self.tournamentSize)
        parents = sorted(parents, key=lambda x: x[1])
        bestparent = parents[0]
        return bestparent

    def crossover(self, matingPoolTours):
        random.shuffle(matingPoolTours)
        crossovers = []
        while len(matingPoolTours) >= 2:
            parent1 = matingPoolTours.pop()
            parent2 = matingPoolTours.pop()
            if random.uniform(0, 1) <= 0.8:
                (child1, child2) = self.OX(parent1, parent2)
            else:
                (child1, child2) = self.PMX(parent1, parent2)
            crossovers.append(child1)
            crossovers.append(child2)
        return crossovers + matingPoolTours  # Returns the new children + the leftover parents

    def mutate(self, crossovers, distanceMatrix):
        results = []
        for i in range(len(crossovers)):  # 0 ... 3999
            original = crossovers[i]
            if random.uniform(0, 1) <= self.mutationRate:
                swapped = self.invert(original)
                results.append(swapped)
            else:
                results.append(original)
        return results

    def swap(self, tour):
        indeces = list(range(len(tour)))
        [index1, index2] = random.sample(indeces, 2)
        number1 = tour[index1]
        number2 = tour[index2]
        tour[index1] = number2
        tour[index2] = number1
        return tour

    def invert(self, aTour):
        indeces = list(range(len(aTour)))
        [index1, index2] = random.sample(indeces, 2)
        index1, index2 = min(index1, index2), max(index1, index2)
        aTour[index1:index2 + 1] = aTour[index1: index2 + 1][::-1]
        return aTour

    def eliminate(self, mutations, distanceMatrix):
        population = []
        amount = self.populationSize * (1 - self.elitismRate)
        mutations = mutations[:int(amount)]

        for i in range(len(mutations)):  # this loop appends the length to all the tours
            tour = mutations[i]
            distance = self.length(distanceMatrix, tour)
            population.append((tour, distance))
        # population = population[:int(amount)] 
        return population

    def PMX(self, parent1, parent2):

        zeros = [0] * len(parent1)

        firstCrossPoint = random.randint(0, len(parent1) - 1)
        secondCrossPoint = random.randint(firstCrossPoint + 1, len(parent1))

        parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
        parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

        child1 = (zeros[:firstCrossPoint] + parent1MiddleCross + zeros[secondCrossPoint:])
        child2 = (zeros[:firstCrossPoint] + parent2MiddleCross + zeros[secondCrossPoint:])

        # i1 are the elements that occur in child2 and not in child1
        # i2 are the elements that occur in child1 and not in child2
        i1 = []  # i1 = [8,2]
        i2 = []  # i2 = [4,7]
        for k in range(len(parent2MiddleCross)):
            if parent2MiddleCross[k] not in parent1MiddleCross:
                i1.append(parent2MiddleCross[k])
            if parent1MiddleCross[k] not in parent2MiddleCross:
                i2.append(parent1MiddleCross[k])

        # j1 are the elements of child1 that are on the same position of the elements of i1 in child2
        # j2 are the elements of child2 that are on the same position of the elements of i1 in child1
        j1 = []  # j1 = [4,5]
        j2 = []  # j2 = [8,5]
        for k in range(len(i1)):
            index = parent2.index(i1[k])
            j1.append(child1[index])

        for k in range(len(i2)):
            index = parent1.index(i2[k])
            j2.append(child2[index])

        for k in range(len(i1)):
            index = parent2.index(j1[k])
            number = child1[index]
            while number != 0:
                index = parent2.index(number)
                number = child1[index]
            child1[index] = i1[k]

        for k in range(len(i2)):
            index = parent1.index(j2[k])
            number = child2[index]
            while number != 0:
                index = parent1.index(number)
                number = child2[index]
            child2[index] = i2[k]

        for n in range(len(child1)):  # 0 .. 8
            if child1[n] == 0:
                child1[n] = parent2[n]
            if child2[n] == 0:
                child2[n] = parent1[n]

        return (child1, child2)

    def OX(self, parent1, parent2):
        firstCrossPoint = random.randint(0, len(parent1) - 1)
        secondCrossPoint = random.randint(firstCrossPoint + 1, len(parent1))
        zeros = [0] * len(parent1)
        parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
        parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]
        child1 = (zeros[:firstCrossPoint] + parent1MiddleCross + zeros[secondCrossPoint:])
        child2 = (zeros[:firstCrossPoint] + parent2MiddleCross + zeros[secondCrossPoint:])
        num = 0
        num2 = 0
        for i in (parent2[secondCrossPoint:] + parent2[:secondCrossPoint]):
            if i not in child1:
                if (secondCrossPoint + num) >= len(child1):  # if going over, append from the beginning
                    child1[num2] = i
                    num2 += 1
                else:
                    child1[secondCrossPoint + num] = i
                    num += 1

        num = 0
        num2 = 0
        for i in (parent1[secondCrossPoint:] + parent1[:secondCrossPoint]):
            if i not in child2:
                if (secondCrossPoint + num) >= len(child2):
                    child2[num2] = i
                    num2 += 1
                else:
                    child2[secondCrossPoint + num] = i
                    num += 1
        return (child1, child2)

    def calculateMean(self, population):
        sum = 0
        length = len(population)
        for i in range(length):
            sum += population[i][1]
        return sum / length




