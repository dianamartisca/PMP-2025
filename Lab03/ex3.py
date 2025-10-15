import random
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from scipy.stats import binom

#a
def simulate_game():
    starting_player = random.choice(["P0", "P1"])

    n = random.randint(1, 6)  

    if starting_player == "P0":
        m = sum(1 for _ in range(2 * n) if random.random() < 4/7)
    else:
        m = sum(1 for _ in range(2 * n) if random.random() < 0.5)
    
    if n >= m:
        return starting_player  
    else:
        return "P1" if starting_player == "P0" else "P0"

def simulate_games(num_games):
    wins = {"P0": 0, "P1": 0}
    
    for _ in range(num_games):
        winner = simulate_game()
        wins[winner] += 1
    
    p0_win_prob = wins["P0"] / num_games
    p1_win_prob = wins["P1"] / num_games
    
    return p0_win_prob, p1_win_prob

p0_win_prob, p1_win_prob = simulate_games(10000)

print(f"Probability of P0 winning: {p0_win_prob:.4f}")
print(f"Probability of P1 winning: {p1_win_prob:.4f}")

#b
model = DiscreteBayesianNetwork([
    ('C', 'N'),  
    ('C', 'M'),  
    ('N', 'M'),  
    ('N', 'W'),  
    ('M', 'W')   
])

cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.5], [0.5]])

cpd_n = TabularCPD(
    variable='N', 
    variable_card=6, 
    values=[
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
        [1/6, 1/6],
    ], 
    evidence=['C'], 
    evidence_card=[2]
)

cpd_m = TabularCPD(
    variable='M', 
    variable_card=13,  
    values=[
        [binom.pmf(m, 2 * n, 0.5) if c == 0 else binom.pmf(m, 2 * n, 4/7)
         for c in range(2) for n in range(1, 7)]
        for m in range(13)
    ],
    evidence=['C', 'N'],
    evidence_card=[2, 6]
)

cpd_w = TabularCPD(
    variable='W', variable_card=2,
    values=[
        # P0 wins if N >= M
        [1 if n >= m else 0 for n in range(1, 7) for m in range(13)],
        # P1 wins if N < M
        [0 if n >= m else 1 for n in range(1, 7) for m in range(13)]
    ],
    evidence=['N', 'M'],
    evidence_card=[6, 13]
)

model.add_cpds(cpd_c, cpd_n, cpd_m, cpd_w)

assert model.check_model()

inference = VariableElimination(model)

result = inference.query(variables=['W'])
print("Probability of winning:")
print(result)

#c
result_c = inference.query(variables=['C'], evidence={'M': 1})

print("Posterior probability of the starting player given M=1:")
print(result_c)

