# Inicialização - Opção usando código na mão

# Aqui pode ser uma classe
import numpy as np
import random

rng = np.random.default_rng()

class GA:
    def __init__(self):
        pass


    def start_population(max_size):
        '''
        Inicialização
        Criação aleatória de um conjunto de soluções candidatas (população inicial).
        **Aqui vamos criar individuos de forma aleatória para cada um dos modelos que serão utilizados.**

        20 Individuos com espaco de busca de 3 dimensões (3 modelos), 
        de pesos continuos, com valores entre 0 e 1, e a soma dos 
        pesos deve ser igual a 1.

        # # Opção usando NumPy - dirichlet
        # rng = np.random.default_rng()
        # rng.dirichlet([1, 1, 1], size=20)
        
        '''
        population = []
        epsilon = 1e-6

        for _ in range(max_size):

            ind = []

            first_element = random.uniform(epsilon, 0.98)
            second_element = random.uniform(epsilon, 1 - first_element - epsilon)
            third_element = 1 - (first_element + second_element)

            ind = [first_element, second_element, third_element]
            random.shuffle(ind)

            population.append(ind)

        return np.array(population)

    def evaluate_population(y_test, y_hat):

        '''
        2. Avaliação
        
        Aplica-se uma função de fitness no conjunto.
        Essa função define o quão boa é cada solução (quanto maior ou menor, melhor — depende do problema).
        Nesse processo, os individuos inicializados serão executados em cada modelo e avaliados de acordo com o resultado do fitness utilizado.
        
        '''
        ind_metrics = []

        for _ in range(y_hat.shape[1]):
            fitness = 1 / mean_squared_error(y_test, y_hat[:,_])
            ind_metrics.append(fitness)

        return ind_metrics

    def tournament_selection(population, fitness, size_tournament):
        '''
        # 3. Seleção
        # Alguns dos melhores candidatos são escolhidos.
        # Esses candidatos irão gerar a próxima geração.

        # Distinguir com base na qualidade, permitir que os melhores individuos se tornem pais da próxima geração.

        Processo probabilístico, onde indivíduos com melhor fitness têm maior chance de serem selecionados, mas mesmo os piores têm uma chance (evitando convergência prematura).
        
        '''
        n_population = len( population )
        selected = []

        for _ in range(n_population):
            participants = random.sample( range(len(population)), size_tournament )

            winner = participants[0]
            for i in range(1, size_tournament):
                if fitness[participants[i]] > fitness[winner]:
                    winner = participants[i]
            
            selected.append(population[winner])

        return selected


    def crossover(selected_parents):
        '''
        Cruzamento (Recombinação)
        
        - A partir de dois ou mais candidatos (pais), são gerados novos indivíduos (filhos). 
        '''
        new_generation = []

        for gen in range(0, len(selected_parents), 2):
            parent_1, parent_2 = selected_parents[gen], selected_parents[gen+1]
            single_point = random.randint(1, len(parent_1) - 1)

            child_1 = np.concatenate((parent_1[:single_point], parent_2[single_point:]))
            child_2 = np.concatenate((parent_2[:single_point], parent_1[single_point:]))
            
            new_generation.extend([child_1, child_2])
        
        return new_generation

    def mutation(offspring):
        '''
        Mutação
        - Pequenas alterações aleatórias são introduzidas em alguns indivíduos.
        - Isso ajuda a manter a diversidade genética e evita que o algoritmo fique preso em ótimos locais.

        - Se usarmos child[gene] = random.uniform(0, 1) - Dessa forma eu mudo completamente o Gene, o que teoricamente favorece Exploração, comportamentoo mais agfressivo, diversidade e menos estabilidade

        - Se usarmos child[gene] += random.uniform(-0.001, 0.001) - dessa forma eu favoreço refinamento local, busca gradual, comportamento mais estável, e exploração fina.

        '''
        mutation_rate = 0.1

        for i, child in enumerate(offspring):
            mutation = random.random()

            if mutation < mutation_rate:

                gene = random.randint(0, len(child) - 1)
                child[gene] += random.uniform(-0.001, 0.001)

                child = np.clip(child, 0, None)
                child = child / np.sum(child) # Renormalização para garantir que a soma dos pesos seja igual a 1

                offspring[i] = child
        
        return np.array(offspring)

    def renormalizar():
        """
        # Normalização para garantir que a soma dos pesos seja igual a 1?
        # new = child_1 / sum(child_1) 
        De acordo com Domingos, nesse passo eu devo - Garantir que os somatórios dos pesos sejam iguais a 1
        """
        pass

    def genetic_algorithm(P, y_test):
        '''
        '''

        W = start_population(20)
        best_ = []

        for generation in range(50):
            y_hat = P @ W.T
            fitness = evaluate_population(y_test, y_hat)
            best_.append((max(fitness)))


            selected = tournament_selection(W, fitness, 3)
            new_generation = crossover(selected)
            W = mutation(new_generation)
        
        y_hat = P @ W.T
        fitness = evaluate_population(y_test, y_hat)
        c

        best_ind = W[best_idx]
        best_fit = fitness[best_idx]

        # best_ind = max(y_hat, key=evaluate_population(y_test, y_hat))
        # fitness = evaluate_population(y_test, y_hat)

        return best_, best_ind, best_fit

    