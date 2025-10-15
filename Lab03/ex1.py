from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product

#a
model = DiscreteBayesianNetwork([
    ('S', 'O'),  
    ('S', 'L'), 
    ('S', 'M'),  
    ('L', 'M')   
])

print("Independencies in the Bayesian Network:")
print(model.get_independencies())

#b
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.4], [0.6]])
cpd_o = TabularCPD(variable='O', variable_card=2, 
                   values=[[0.7, 0.1], [0.3, 0.9]], 
                   evidence=['S'], evidence_card=[2])
cpd_l = TabularCPD(variable='L', variable_card=2, 
                   values=[[0.8, 0.3], [0.2, 0.7]], 
                   evidence=['S'], evidence_card=[2])
cpd_m = TabularCPD(variable='M', variable_card=2, 
                   values=[[0.9, 0.5, 0.6, 0.2], [0.1, 0.5, 0.4, 0.8]], 
                   evidence=['S', 'L'], evidence_card=[2, 2])

model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)

assert model.check_model()

inference = VariableElimination(model)

combinations = list(product([0, 1], repeat=3))

for o, l, m in combinations:
    evidence = {'O': o, 'L': l, 'M': m}
    result = inference.query(variables=['S'], evidence=evidence)
    print(f"Evidence: {evidence}")
    print(result)

