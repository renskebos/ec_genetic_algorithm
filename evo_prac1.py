import random
import time
import numpy as np
import matplotlib.pyplot as plt


class individual:
   # Initialize a new individual object with bitstring and fitness
   def __init__(self, bitstring=None):
       self.bitstring = bitstring if bitstring else self.generate_bitstring()
       self.fitness = None


   @staticmethod
   def generate_bitstring(length=40):
       # Generate random bitstring with length 40
       return format(random.getrandbits(length), f'0{length}b')


   # Evaluate and set the fitness of individual using fitness function
   def set_fitness(self, fitness):
           self.fitness = fitness


###
# The class for genetic algorithm
###


class geneticAlgorithm:
   def __init__(self, population_size, crossover_type, fitness_function_type):
       # Set population size, crossover type and fitness function type
       self.population_size = population_size
       self.crossover_type = crossover_type
       self.population = [individual() for _ in range(population_size)]
        # set stopping criteria and other measurements
       self.stopping = False
       self.optimum = False
       self.best_bitstring = []
       self.generations_since_improvement = 0 # Track generations without improvements
       self.fitness_evaluations = 0
       self.cpu_time_start = 0
       self.total_cpu_time = 0
       # Assign the correct fitness based on type
       self.fitness_function = self.get_fitness_function(fitness_function_type)
      
   def get_fitness_function(self, fitness_function_type):
       # Fitness function mapping
       fitness_functions = {
           'counting_ones': counting_ones_function,
           'deceptive_trap': deceptive_trap_function,
           'non_deceptive_trap': non_deceptive_trap_function,
           'non_tightly_linked_trap':  not_tightly_linked_trap_function,
           'non_tightly_linked_non_deceptive_trap': not_tightly_linked_non_deceptive_trap_function
       }
       return fitness_functions.get(fitness_function_type)
  
   # Shuffle population to randomize parent selection process
   def shuffle_and_pair(self):
       random.shuffle(self.population)
       # Handle method when population size is odd
       pair_limit = len(self.population) - (len(self.population) % 2)
       return [(self.population[i], self.population[i + 1]) for i in range(0, pair_limit, 2)]




###
# Different types of crossover operators
###


   # Decide which type of crossover: uniform or two-point
   def crossover(self, parent1, parent2):
       if self.crossover_type == 'uniform':
           return self.uniform_crossover(parent1, parent2)
       elif self.crossover_type == 'twopoint':
           return self.twopoint_crossover(parent1, parent2)
     
   @staticmethod
   def uniform_crossover(parent1, parent2):
       # Uniform crossover: coin flip for each bit
       offspring1 = ''.join([parent1[i] if random.choice([True, False]) else parent2[i] for i in range(len(parent1))])
       offspring2 = ''.join([parent2[i] if random.choice([True, False]) else parent1[i] for i in range(len(parent1))])
       return offspring1, offspring2


   @staticmethod
   def twopoint_crossover(parent1, parent2):
       # Two-point crossover: bits in between two points are swapped between the parents
       point1, point2 = sorted(random.sample(range(len(parent1)), 2))
       offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
       offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
       return offspring1, offspring2


###
# Family competition
###


   def family_competition(self, parents, offspring):
        family = parents + offspring
       # Select the fittest individuals to form next generation by evaluating the fitness of each individual in the population
        for ind in family:
            if ind.fitness == None:
                self.fitness_evaluations += 1
                ind.set_fitness(self.fitness_function(ind.bitstring))
       # Sort family members by fitness in descending order. Chooses offspring over parents in case of equal fitness
        family_sorted = sorted(family, key=lambda x: (-x.fitness, family.index(x) >= len(parents)))
       # Select 2 top family members
        return family_sorted[:2]


###
# Run for number of generations
###


   def run_generation(self):
       # Run the GA for a number of generations
       new_population = []
       best_fitness = []


       #Pair off individuals for crossover
       for parent1, parent2 in self.shuffle_and_pair():
           offspring1_bitstring, offspring2_bitstring = self.crossover(parent1.bitstring, parent2.bitstring)
           # Create individual objects for offspring
           offspring1 = individual(offspring1_bitstring)
           offspring2 = individual(offspring2_bitstring)
  
           # Family competition
           best = self.family_competition([parent1, parent2], [offspring1, offspring2])
           best_fitness.append(best[0].fitness)
           self.optimum_found(best[0])
           new_population.extend(best)
     
       # Update generation information and check for stopping criteria
       current_generation_best_fitness = max(best_fitness)
       self.best_bitstring.append(current_generation_best_fitness)  # Track the best fitness in the current generation


       if len(self.best_bitstring) > 10:  # Ensure there are enough generations to compare
           self.stopping_criteria(current_generation_best_fitness)  # Check if the stopping criteria are met based on the history of best fitnesses


       self.population = new_population  # Update the population for the next generation
 
   def optimum_found(self, best_member):
       # Return True if need to stop with making babiezzz
       if best_member.fitness == 40:
           self.optimum = True
           self.stopping = True # Stop the algorithm if the global optimum if found
     
   def stopping_criteria(self, fittest_individual):
       # Compare the best fitness of the current generation with the best fitnesses of the previous 10 generations
        if fittest_individual <= min(self.best_bitstring[-11:-1]):
            self.stopping = True  # No improvement over the last 10 generations, so stop the algorithm


   def run(self):
       # Run the GA until optimum found or no increase in fitness
       self.generations = 0
       self.cpu_time_start = time.time() # Start timing
       while not self.stopping and not self.optimum:
           self.generations += 1
           self.run_generation()


       # At the end of the run method: calculate cpu time and select best individual
       self.total_cpu_time = time.time() - self.cpu_time_start
       best_individual = max(self.population, key=lambda ind: ind.fitness)
       # print(f'Best Individual: {best_individual.bitstring}, Fitness: {best_individual.fitness}, Top fitnesses: {self.best_bitstring}, Population size: {self.population_size}')
      
       return self.optimum, self.generations, self.fitness_evaluations, self.total_cpu_time


###
# Different types of fitness functions
###


def counting_ones_function(binary_string):
   # Function that counts number of '1's in the bitstring
   return binary_string.count('1')




def deceptive_trap_function(binary_string, k=4, d=1):
   # Define a dictionary for mapping number of ones to fitness values
   fitness_mapping = {4: 4, 3: 0, 2: 1, 1: 2, 0: 3}


   # Split string into parts of k bits and calculate fitness value
   return sum(fitness_mapping[binary_string[i:i+k].count('1')] for i in range(0, len(binary_string), k))




def non_deceptive_trap_function(binary_string, k=4, d=2.5):
   # Define a dictionary for mapping number of ones to fitness values
   fitness_mapping = {4: 4, 3: 0, 2: 0.5, 1: 1, 0: 1.5}
 
   # Split string into parts of k bits and calculate fitness value
   return sum(fitness_mapping[binary_string[i:i+k].count('1')] for i in range(0, len(binary_string), k)) 




def not_tightly_linked_trap_function(binary_string):
   # Define a dictionary for mapping number of ones to fitness values
   fitness_mapping = {4: 4, 3: 0, 2: 1, 1: 2, 0: 3}


   # Calculate fitness value based on the mapping
   total_fitness = 0
   for i in range(10):
       part = binary_string[i] + binary_string[i+10] + binary_string[i+20] + binary_string[i+30]
       num_ones = part.count('1')
       total_fitness += fitness_mapping[num_ones]
   return total_fitness




def not_tightly_linked_non_deceptive_trap_function(binary_string):
   # Define a dictionary for mapping number of ones to fitness values
   fitness_mapping = {4: 4, 3: 0, 2: 0.5, 1: 1, 0: 1.5}


   # Calculate fitness value based on the mapping
   total_fitness = 0
   for i in range(10):
       part = binary_string[i] + binary_string[i+10] + binary_string[i+20] + binary_string[i+30]
       num_ones = part.count('1')
       total_fitness += fitness_mapping[num_ones]
   return total_fitness


###
# Run experiment + find minimal population size
###
def run_experiment(population_size, crossover_type, fitness_function_type, total_runs=20):
   success_count = 0
   total_runs = 20
   generations = []
   evaluations = []
   cpu_times =[]
  
   for _ in range(total_runs):
       ga = geneticAlgorithm(population_size=population_size, crossover_type=crossover_type, fitness_function_type=fitness_function_type)
       result = ga.run()
       if result[0]: # Check if run successful
           success_count += 1
           generations.append(result[1])
           evaluations.append(result[2])
           cpu_times.append(result[3])


   return success_count, generations, evaluations, cpu_times


def find_min_popsize(fitness_function_type, crossover_type, total_runs):
   lower_bound = 10
   upper_bound = None
   reliable_solution_found = False
   all_generations, all_evaluations, all_cpu_times = [], [], []
  


   # Increase until success or max population size is reached
   N = 10
   tested_pop_sizes = [10]
   while N <= 1280 and not reliable_solution_found:
       success_count, generations, evaluations, cpu_times = run_experiment(tested_pop_sizes[-1], crossover_type, fitness_function_type, total_runs)
       if success_count >= 19:  # Check if the problem is solved reliably
           reliable_solution_found = True
           upper_bound = tested_pop_sizes[-1]
           lower_bound = tested_pop_sizes[-2]
           # Select data from successful experiment
           all_generations.extend(generations)
           all_evaluations.extend(evaluations)
           all_cpu_times.extend(cpu_times)
       else:
           N *= 2 #Double the population
           tested_pop_sizes.append(N)
  
   if not reliable_solution_found:
       print("Failed to find reliable solution found within the given population size limits.")
       return


   # Bisection search
   while abs(lower_bound - upper_bound) > 10:
       N = (lower_bound + upper_bound) // 2
       tested_pop_sizes.append(N)
       success_count, generations, evaluations, cpu_times = run_experiment(N, crossover_type, fitness_function_type, total_runs)

       if success_count >= 19:
           upper_bound = N
           # Select data from successful experiment
           all_generations.extend(generations)
           all_evaluations.extend(evaluations)
           all_cpu_times.extend(cpu_times)
       else:
           lower_bound = N
  
   print(f'All population sizes: {tested_pop_sizes}')
   print(f'Minimal successful population size found: {upper_bound}')


   # Final reporting       
   if reliable_solution_found and all_generations:
       print(f"Minimal population size for reliable solution: {upper_bound}")
       print(f'Average generations: {np.mean(generations)}, (Std Dev: {np.std(generations)})')
       print(f'Average fitness evaluations: {np.mean(evaluations)}, (Std Dev: {np.std(evaluations)})')
       print(f'Average CPU time (seconds): {np.mean(cpu_times)}, (Std Dev: {np.std(cpu_times)})')
   else:
       print("Failed to find a reliable solution even with large population sizes.")


###
# Main
###        


if __name__ == "__main__":
# Define all experiments
   experiments = [
       ('counting_ones', 'uniform'),
       ('counting_ones', 'twopoint'),
       ('deceptive_trap', 'uniform'),
       ('deceptive_trap', 'twopoint') ,
       ('non_deceptive_trap', 'uniform'),
       ('non_deceptive_trap', 'twopoint'),
       ('non_tightly_linked_trap', 'uniform'),
       ('non_tightly_linked_trap', 'twopoint'),
       ('non_tightly_linked_non_deceptive_trap', 'uniform'),
       ('non_tightly_linked_non_deceptive_trap', 'twopoint')
   ]


   # Define total runs before entering the loop
   total_runs = 20


   # Loop through all experiments
   for fitness_function_type, crossover_type in experiments:  
       print(f"\nExperiment: {fitness_function_type} with {crossover_type} crossover")
       find_min_popsize(fitness_function_type, crossover_type, total_runs)



