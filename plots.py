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
       # Record proportions of bits-1
       self.prop_t = []
       self.selection_errors = {}
       self.correct_selections = {}
       self.bit_with_one = {}
       self.bit_with_zero = {}
       self.fitness_one_bits = {}
       self.fitness_zero_bits ={}
       self.std_one_bit = {}
       self.std_zero_bit = {}
       one_bits = 0
       zero_bits = 0
       list_fitness0 =[]
       list_fitness1=[]
       # Count the solutions in each schemata
       for ind in self.population:
        if ind.bitstring[0] == '1':
            one_bits += 1
            schema_1_fitness = self.fitness_function(ind.bitstring)
            list_fitness1.append(schema_1_fitness)
        elif ind.bitstring[0] == '0':
            zero_bits += 1
            schema_0_fitness = self.fitness_function(ind.bitstring)
            list_fitness0.append(schema_0_fitness)
            # Save the counts for plotting
        self.bit_with_one[0] = one_bits
        self.bit_with_zero[0] = zero_bits

        # Calculate average fitness and standard deviation for '0' bits
        avg_fitness_0 = np.mean(list_fitness0) if list_fitness0 else 0
        std_0 = np.std(list_fitness0) if list_fitness0 else 0

        avg_fitness_1 = np.mean(list_fitness1) if list_fitness1 else 0
        std_1 = np.std(list_fitness1) if list_fitness1 else 0
        
        # Save the results for plotting
        self.fitness_one_bits[0] = avg_fitness_1
        self.fitness_zero_bits[0] = avg_fitness_0
        self.std_one_bit[0] = std_1
        self.std_zero_bit[0] = std_0


   def get_fitness_function(self, fitness_function_type):
       # Fitness function mapping
       fitness_functions = {
           'counting_ones': counting_ones_function,
           'deceptive_trap': deceptive_trap_function,
           'non_deceptive_trap': non_deceptive_trap_function,
           'not_tightly_linked_trap':  not_tightly_linked_trap_function,
           'not_tightly_linked_non_deceptive_trap': not_tightly_linked_non_deceptive_trap_function
       }
       return fitness_functions.get(fitness_function_type)
  
   def evaluate_population_fitness(self):
       for ind in self.population:
           # Increment counter and evaluate fitness
           self.fitness_evaluations += 1
           fitness = self.fitness_function(ind.bitstring)
           ind.set_fitness(fitness)
  
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
       errors = 0
       correct_selections = 0
       # Select the most fittest individuals to form next generation by evaluating the fitness of each individual in the population
       for ind in family:
           self.fitness_evaluations += 1
           ind.set_fitness(self.fitness_function(ind.bitstring))
       # Sort family members by fitness in descending order. Chooses offspring over parents in case of equal fitness
       family_sorted = sorted(family, key=lambda x: (-x.fitness, family.index(x) >= len(parents)))
       # Select 2 top family members

       for i in range(40):
            if family[0].bitstring[i] != family[1].bitstring[i] and family_sorted[0].bitstring[i] == '0' and family_sorted[1].bitstring[i]== '0':
                errors += 1
                
            elif family[0].bitstring[i] != family[1].bitstring[i] and family_sorted[0].bitstring[i] == '1' and family_sorted[1].bitstring[i]== '1':
                correct_selections += 1

       self.update_number_of_errors(errors, correct_selections)
       return family_sorted[:2]
   
   def update_number_of_errors(self, errors, correct_selections):
        # Update dictionaries with cumulative sums
        self.selection_errors[self.generations] = self.selection_errors.get(self.generations, 0) + errors
        self.correct_selections[self.generations] = self.correct_selections.get(self.generations, 0) + correct_selections
        return(self.selection_errors, self.correct_selections)

###
# Run for number of generations
###


   def run_generation(self):
       # Run the GA for a number of generations
       new_population = []
       best_fitness = []
       list_fitness0 =[]
       list_fitness1=[]
       # Initialize counts for schemata 1 and 0
       one_bits = 0
       zero_bits = 0
       # Count the solutions in each schemata
       for ind in self.population:
        if ind.bitstring[0] == '1':
            schema_1_fitness = self.fitness_function(ind.bitstring)
            one_bits += 1
            list_fitness1.append(schema_1_fitness)
        elif ind.bitstring[0] == '0':
            schema_0_fitness = self.fitness_function(ind.bitstring)
            zero_bits += 1
            list_fitness0.append(schema_0_fitness)
    
            # Save the counts for plotting
        self.bit_with_one[self.generations] = one_bits
        self.bit_with_zero[self.generations] = zero_bits

        # Calculate average fitness and standard deviation for '0' bits
        avg_fitness_0 = np.mean(list_fitness0) if list_fitness0 else 0
        std_0 = np.std(list_fitness0) if list_fitness0 else 0

        avg_fitness_1 = np.mean(list_fitness1) if list_fitness1 else 0
        std_1 = np.std(list_fitness1) if list_fitness1 else 0
        
        # Save the results for plotting
        self.fitness_one_bits[self.generations] = avg_fitness_1
        self.fitness_zero_bits[self.generations] = avg_fitness_0
        self.std_one_bit[self.generations] = std_1
        self.std_zero_bit[self.generations] = std_0
        
       #Pair off individuals for crossover
       for parent1, parent2 in self.shuffle_and_pair():
           offspring1_bitstring, offspring2_bitstring = self.crossover(parent1.bitstring, parent2.bitstring)
           # Create individual objects for offspring
           offspring1 = individual(offspring1_bitstring)
           offspring2 = individual(offspring2_bitstring)
           offspring1.set_fitness(self.fitness_function(offspring1.bitstring))
           offspring2.set_fitness(self.fitness_function(offspring2.bitstring))


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


       self.prop_t.append(self.calculate_prop_t())


   def optimum_found(self, best_member):
       # Return True if need to stop with making babiezzz
       if best_member.fitness == 40:
           self.optimum = True
           self.stopping = True # Stop the algorithm if the global optimum if found
     
   def stopping_criteria(self, fittest_individual):
       # Check if there has been no improvement over the last 10 generations
       if len(self.best_bitstring) >= 11:  # Ensure there's enough history to check
       # Compare the best fitness of the current generation with the best fitnesses of the previous 10 generations
           if fittest_individual <= min(self.best_bitstring[-11:-1]):
               self.stopping = True  # No improvement over the last 10 generations, so stop the algorithm


   def calculate_prop_t(self):
       # Calcculate the propportion of bits-1 in the entire population
       total_bits = len(self.population) * len(self.population[0].bitstring)
       total_ones = sum(ind.bitstring.count('1') for ind in self.population)
       return total_ones / total_bits


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
       print(f'Average generations: {np.mean(all_generations)}, (Std Dev: {np.std(all_generations)})')
       print(f'Average fitness evaluations: {np.mean(all_evaluations)}, (Std Dev: {np.std(all_evaluations)})')
       print(f'Average CPU time (seconds): {np.mean(all_cpu_times)}, (Std Dev: {np.std(all_cpu_times)})')
   else:
       print("Failed to find a reliable solution even with large population sizes.")


###
# Plot functions
###
def plot_proportions(prop_t, filename= 'plot.png', title='Proportion of bits-1 in the population over generations', xlabel='Generation', ylabel='Proportion of bits-1', ymin=0, ymax=1):
   plt.figure(figsize=(10, 6))  
   plt.plot(prop_t)
   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.xlim(0, len(prop_t) - 1)
   plt.ylim(ymin, ymax)
   plt.grid(True)
   # Save plot to the file
  # Specify the directory path
   directory_path = "/Users/renskebos/Documents/AI_Master/evolutionary computing/plots"
  
   # Save plot to the file with the specified filename
   full_path = directory_path + filename
   plt.savefig(full_path)


   # Display the plot
   plt.show()


def plot_errors(generation_errors, generation_correct_selections):
    plt.figure(figsize=(10, 6))

    if generation_errors:
        generations, combined_errors = zip(*generation_errors.items())
        plt.plot(generations, combined_errors, label='Selection Errors', linestyle='-')

    if generation_correct_selections:
        generations, combined_correct_selections = zip(*generation_correct_selections.items())
        plt.plot(generations, combined_correct_selections, label='Correct Selections', linestyle='-')

    plt.title('Selection Errors and Correct Selections per generation')
    plt.xlabel('Generation')
    plt.ylabel('Number of selections')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/renskebos/Documents/AI_Master/evolutionary computing/plots')
    plt.show()

def plot_one_bits(bit_with_one, bit_with_zero):

    generations_1, counts_1 = zip(*bit_with_one.items())
    generations_0, counts_0 = zip(*bit_with_zero.items())

    plt.plot(generations_1, counts_1, label='1**..** schemata', linestyle='-')
    plt.plot(generations_0, counts_0, label='0**..** schemata', linestyle='-')

    plt.title('Number of solutions of schemata')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('/Users/renskebos/Documents/AI_Master/evolutionary computing/one_bit')
    plt.show()

# Function to plot fitness and standard deviation as a function of the generation
def plot_fitness_and_std(generations, fitness_one_bits, fitness_zero_bits, std_one_bit, std_zero_bit):
    plt.figure(figsize=(10, 6))

    # Plot fitness values
    plt.plot(generations, list(fitness_one_bits.values()), label='Fitness of 1**..** schemata', linestyle='-')
    plt.plot(generations, list(fitness_zero_bits.values()), label='Fitness of 0**..** schemata', linestyle='-')

    # Plot standard deviation values
    plt.plot(generations, list(std_one_bit.values()), label='Std of 1**..** schemata', linestyle='--')
    plt.plot(generations, list(std_zero_bit.values()), label='Std of 0**..** schemata', linestyle='--')

    plt.title('Fitness and Standard Deviation over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness / Standard Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
   # Setup for the specific task of optimizing the Counting Ones function
   population_size = 200
   crossover_type = 'uniform'  # Assuming uniform crossover as per requirement
   fitness_function_type = 'counting_ones'
  
   # Initialize and run the GA with specified parameters
   ga = geneticAlgorithm(population_size=population_size, crossover_type=crossover_type, fitness_function_type=fitness_function_type)
   ga.run()  # Run the genetic algorithm


# Plot the generation-wise errors and correct selections in one plot
   #plot_generation_errors(ga.selection_errors, ga.correct_selections)

   print("Number of generations:", len(ga.prop_t))


   # Plot the proportions of bits-1 in the population over generations
   #plot_proportions(ga.prop_t, filename='proportion_of_bits-1_over_generations.png')
   #plot_errors(ga.selection_errors, ga.correct_selections)
   #plot_one_bits(ga.bit_with_one, ga.bit_with_zero)
   plot_fitness_and_std(ga.fitness_one_bits.keys(), ga.fitness_one_bits, ga.fitness_zero_bits, ga.std_one_bit, ga.std_zero_bit)





