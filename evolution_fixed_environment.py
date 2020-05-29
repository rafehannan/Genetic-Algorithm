import random
import numpy as np
import matplotlib.pyplot as plt

B = 30
c = 0.9


# These are just some lists that we save information in for later use
class save_point:
    firm = []
    firm_after_crossover = []
    firm_pre_election = []
    fitness_firm = []
    fitness_firm_pre_election = []
    roulette = []
    prices_from_selection = []
    prices_from_selection_negative_one = []
    price = []


# The bulk of the genetic algorithm is in this class
class main:
    def string(length):
        strings = []
        while len(strings) < length:
            strings.append(random.randint(0, 1))
        return strings

    def firm_gen(pool, M, genes):
        for i in range(pool):
            temp_agents = []
            while len(temp_agents) < M:
                agents = main.string(genes)
                temp_agents.append(agents)
            save_point.firm.append(temp_agents)
        return save_point.firm

    def mutation(mutation_probability):
        mutated_agents = []
        for i in save_point.firm_after_crossover:
            new_agents = []
            for x in i:
                temp_firm = []
                for y in x:
                    chance = random.uniform(0, 1)
                    if chance > mutation_probability:
                        temp_firm.append(y)
                    elif chance < mutation_probability and y == 0:
                        y = 1
                        temp_firm.append(y)
                    elif chance < mutation_probability and y == 1:
                        y = 0
                        temp_firm.append(y)
                new_agents.append(temp_firm)
            mutated_agents.append(new_agents)
        save_point.firm_pre_election = mutated_agents
        return None

    def crossover(probability, strings):
        offspring = []
        for i in range(len(save_point.firm)):
            temp_offspring = []
            for x in range(len(save_point.firm[i])):
                chance = random.uniform(0, 1)
                if chance < probability:
                    X_chromosome = save_point.firm[i][x][:int(strings / 2)]
                    Y_chromosome = save_point.firm[i][x][int(strings / 2):]
                    K = random.randint(1, strings / 2)
                    U = X_chromosome[K:] + Y_chromosome[:K]
                    V = Y_chromosome[K:] + X_chromosome[:K]
                    Z = U + V
                    temp_offspring.append(Z)
                else:
                    temp_offspring.append(save_point.firm[i][x])
            offspring.append(temp_offspring)
        save_point.firm_after_crossover = offspring
        return None

    def election():
        t_pop = []
        l = 0
        for i, j in zip(save_point.fitness_firm, save_point.fitness_firm_pre_election):
            x_pop = []
            m = 0
            for p, q in zip(i, j):
                if p > q:
                    x_pop.append(save_point.firm[l][m])
                    m += 1
                elif p == q:
                    x_pop.append(save_point.firm[l][m])
                    m += 1
                else:
                    x_pop.append(save_point.firm_pre_election[l][m])
                    m += 1
            t_pop.append(x_pop)
            l += 1
        save_point.firm = t_pop
        return None

    def fitness_pre(B, c):
        save_point.fitness_firm = []
        l = 0
        for i in schedule.individual_price_observations:
            m = 0
            temp_list = []
            for j in i:
                fit = 100 - c * (B - schedule.individual_price_observations[l][m]) ** 2
                if fit > 0:
                    temp_list.append(fit)
                    m += 1
                elif fit < 0:
                    temp_list.append(0.0001)
                    m += 1
                else:
                    temp_list.append(0.0001)
                    m += 1
            l += 1
            save_point.fitness_firm.append(temp_list)
        total_list = []
        for i in save_point.fitness_firm:
            total = 0
            for j in i:
                total += j
            total_list.append(total)
        save_point.roulette = []
        q = 0
        for i in save_point.fitness_firm:
            roulette_temp = []
            for j in i:
                percentage = j / total_list[q]
                roulette_temp.append(percentage)
            q += 1
            save_point.roulette.append(roulette_temp)
        return None

    def fitness_post(B, c):
        save_point.fitness_firm_pre_election = []
        m = 0
        for i in schedule.individual_price_observations:
            k = 0
            temp_list = []
            for j in i:
                fit = 100 - c * (B - schedule.individual_price_observations[m][k]) ** 2
                if fit > 0:
                    temp_list.append(fit)
                    k += 1
                elif fit < 0:
                    temp_list.append(0.0001)
                    k += 1
                else:
                    temp_list.append(0.0001)
                    k += 1
            save_point.fitness_firm_pre_election.append(temp_list)
            m += 1
        return None

    def reproduction(M):
        new_firm = []
        n = 0
        for i in save_point.roulette:
            holder_firm = []
            while len(holder_firm) < M:
                x = np.array(i)
                selection = np.random.choice(range(x.size), p=i)
                holder_firm.append(save_point.firm[n][selection])
            new_firm.append(holder_firm)
            n += 1
        save_point.firm = new_firm
        return None


# This class primarily handles how we do the decoding of the two halves of the strings, alpha and beta from the paper
class bitcode:
    price_starA = []
    price_starB = []
    alpha_perm = []
    beta_perm = []
    simulation_plot_time = []
    simulation_plot_price = []
    alpha_temp = []
    beta_temp = []
    Beta = []
    alpha = []

    def generate_bitcode(string_length):
        bitcode.price_starA = []
        for i in save_point.firm:
            temp_output = []
            for k in i:
                string = ''
                for y in range(0, int(string_length / 2)):
                    string += str(k[y])
                string2 = int(string, 2)
                temp_output.append(string2)
            bitcode.price_starA.append(temp_output)
            normalized = []
            for k in temp_output:
                norm_output = k / max(temp_output)
                normalized.append(norm_output)
            bitcode.price_starA.append(normalized)
        return bitcode.price_starA

    def generate_bitcode2(string_length):
        bitcode.price_starB = []
        for i in save_point.firm:
            temp_output = []
            for k in i:
                string = ''
                for y in range(20, int(string_length)):
                    string += str(k[y])
                string2 = int(string, 2)
                temp_output.append(string2)
                normalized = []
            for k in temp_output:
                norm_output = k / max(temp_output)

                normalized.append(norm_output)
            bitcode.price_starB.append(normalized)

        return bitcode.price_starB


# This class has the bulk of the environment in it, ie. the supply functions, prices, selection and so on; also, it includes our plot function
class schedule:
    individual_price_observations = []
    pred_price_perm_list = []

    def draw_plot():
        plt.plot(bitcode.simulation_plot_time, schedule.pred_price_perm_list, label="Alpha")
        plt.title("Results", fontsize=20)
        plt.xlabel("Periods")
        plt.ylabel("Alpha")
        plt.legend()
        plt.show()

    def selection(agents):
        save_point.prices_from_selection = []
        k = 0
        for i in save_point.roulette:
            x = np.array(i)
            selection = np.random.choice(range(x.size), p=i)
            save_point.prices_from_selection.append(schedule.individual_price_observations[k][selection])
            bitcode.alpha_perm.append(bitcode.alpha_temp[k][selection])
            bitcode.beta_perm.append(bitcode.beta_temp[k][selection])
            k += 1
        schedule.pred_price_perm_list.append(sum(save_point.prices_from_selection) / agents)
        return None

    def phenotype():
        schedule.individual_price_observations = []
        bitcode.alpha_temp = []
        bitcode.beta_temp = []
        m = 0
        for i, j in zip(bitcode.price_starA, bitcode.price_starB):
            k = 0
            temp_list = []
            alpha_list = []
            beta_list = []
            for p, q in zip(i, j):
                phenotype = bitcode.price_starA[m][k] * bitcode.price_starB[m][k] + (1 - bitcode.price_starA[m][k]) * \
                            bitcode.price_starB[m][k]
                temp_list.append(phenotype)
                alpha_list.append(bitcode.price_starA[m][k])
                beta_list.append(bitcode.price_starB[m][k])
                k += 1
            m += 1
            schedule.individual_price_observations.append(temp_list)
            bitcode.alpha_temp.append(alpha_list)
            bitcode.beta_temp.append(beta_list)
        return schedule.individual_price_observations

    def alpha_plot():
        plt.plot(bitcode.simulation_plot_time, bitcode.alpha, label="xt")
        plt.title("xt Values Across Periods", fontsize=20)
        plt.xlabel("Periods")
        plt.ylabel("xt")
        plt.legend()
        plt.show()


# This is the main simulation function that brings everything together
def simulation(period, agents, firm_strats, string_length, crossover_probability, mutation_probability, B, c):
    # I am resetting all of the lists on the first run, so you can run the simulation multiple times!
    save_point.firm = []
    save_point.firm_after_crossover = []
    save_point.firm_pre_election = []
    save_point.fitness_firm = []
    save_point.fitness_firm_pre_election = []
    save_point.roulette = []
    save_point.prices_from_selection = []
    save_point.prices_from_selection_negative_one = []
    save_point.price = []
    schedule.individual_pheno = []
    schedule.individual_price_observations_minus_one = []
    schedule.pred_price_perm_list = []
    bitcode.xt_temp = []
    bitcode.alpha_temp = []
    bitcode.alpha_perm = []
    bitcode.xt_perm = []
    bitcode.simulation_plot_time = []
    bitcode.simulation_plot_price = []
    var_alpha = []
    var_xt = []
    # This sets everything up in period 0
    main.firm_gen(agents, firm_strats, string_length)
    bitcode.generate_bitcode(string_length)
    bitcode.generate_bitcode2(string_length)
    schedule.phenotype()
    print("Initial population:", *save_point.firm, sep="\n")

    # This runs the simulation for t periods
    for i in range(period):
        bitcode.generate_bitcode(string_length)
        bitcode.generate_bitcode2(string_length)
        schedule.phenotype()
        main.fitness_pre(B, c)
        main.reproduction(firm_strats)
        main.crossover(crossover_probability, string_length)
        main.fitness_post(B, c)

        schedule.selection(agents)
        bitcode.simulation_plot_time.append(i)
        bitcode.alpha.append(sum(bitcode.alpha_perm) / len(bitcode.alpha_perm))
        if i % 5 == 0:
            print(bitcode.alpha)
    # Generating a summary of the simulation
    print("Ending agents:", *save_point.firm, sep="\n")

    schedule.draw_plot()
    print("Mean of α: \n", round(sum(schedule.pred_price_perm_list) / len(schedule.pred_price_perm_list), 2))
    print("Mean of xt: \n", round(sum(bitcode.alpha_perm) / len(bitcode.alpha_perm), 2))
    print("Variance of α:\n", np.var(schedule.pred_price_perm_list))
    print("Variance of xt: \n", np.var(bitcode.alpha_perm))
    return "Simulation finished"


print("Hello Arthur, Welcome to my simulation. For your ease of use, I have created a small list of instructions \n"
      "If you would like to run a simulation, you can use our simulation function. Simulation() takes 8 arguments. \n"
      "1. The periods you want to run the simulation for; 2. The number of agents you want; 3. The strategies that each agent possesses \n"
      "4. The string length of each strategy; 5. The probability that crossover will occur; 6. The probability that mutation will occur \n"
      "7. The parameter B which denotes the environment (in here it is fixed)\n"
      "8. The parameter c which is the cost of individual learning\n"
      "Thank you for a fun semester \n"
      "Type this in -> simulation(100, 6, 10, 40, 0.6, 0.01, B,c)")
simulation(30, 6, 10, 40, 0.6, 0.01, B,c)