import random
import matplotlib.pyplot as plt

#Задача: По булевой формуле j(р1, р2, …, рn) выяснить,
# существует ли набор истинностных значений переменных а1,а2, …, аn,
# для которого j(а1,а2, …, аn)=1

class gasat:
    def __init__(self, formula, n_vars, pop_size=100, mutation_rate=0.05,
                 elite_size=10, generations=300):
        self.formula = formula
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        self.clauses = self._parse(formula)

    def _parse(self, formula):
        clauses = []
        for clause in formula.replace(' ', '').split('&'):
            literals = []
            for lit in clause.strip('()').split('|'):
                neg = '~' in lit
                var = int(lit.replace('~', '').replace('p', '')) - 1
                literals.append((var, neg))
            clauses.append(literals)
        return clauses

    def _eval_clause(self, clause, assignment):
        return any(assignment[var] != neg for var, neg in clause)

    def fitness(self, assignment):
        return sum(self._eval_clause(clause, assignment) for clause in self.clauses)

    def random_assignment(self):
        return [random.choice([True, False]) for _ in range(self.n_vars)]

    def crossover(self, p1, p2):
        if self.n_vars <= 1:
            return p1[:]
        point = random.randint(1, self.n_vars - 1)
        return p1[:point] + p2[point:]

    def mutate(self, ind):
        return [not bit if random.random() < self.mutation_rate else bit for bit in ind]

    def select(self, population, fitnesses):
        idx = random.sample(range(len(population)), 3)
        return population[max(idx, key=lambda i: fitnesses[i])][:]

    def solve(self, verbose=True):
        population = [self.random_assignment() for _ in range(self.pop_size)]
        best, best_fitness = None, 0
        history = []

        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in population]

            gen_best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best = population[gen_best_idx][:]

                if best_fitness == len(self.clauses):
                    if verbose:
                        print(f"\nрешение найдено на поколении {gen}!")
                    return best, True, history

            history.append(best_fitness / len(self.clauses))

            if verbose and gen % 50 == 0:
                print(f"поколение {gen}: {best_fitness}/{len(self.clauses)} "
                      f"({best_fitness / len(self.clauses) * 100:.1f}%)")

            new_pop = sorted(population, key=self.fitness, reverse=True)[:self.elite_size]
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select(population, fitnesses), self.select(population, fitnesses)
                child = self.mutate(self.crossover(p1, p2))
                new_pop.append(child)
            population = new_pop

        return best, best_fitness == len(self.clauses), history

    def plot(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot([h * 100 for h in history], 'b-', linewidth=2)
        plt.xlabel('поколение')
        plt.ylabel('выполненные клаузы (%)')
        plt.title('сходимость генетического алгоритма для sat')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=100, color='r', linestyle='--', label='решение найдено')
        plt.legend()
        plt.show()

def run_example(formula, n_vars, name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}\nформула: {formula}\n")

    ga = gasat(formula, n_vars, pop_size=100, generations=200)
    solution, satisfiable, history = ga.solve()

    if satisfiable:
        assignment = {f'p{i + 1}': solution[i] for i in range(n_vars)}
        print(f"\nформула выполнима")
        print(f"набор: {assignment}")
    else:
        print(f"\nформула невыполнима (или решение не найдено)")
        
    ga.plot(history)

if __name__ == "__main__":
    run_example("(p1|p2)&(~p1|p2)", 2, "пример 1: выполнимая формула")
    run_example("(p1)&(~p1)", 1, "пример 2: невыполнимая формула")