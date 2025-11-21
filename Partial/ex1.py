from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt

#a
model = DiscreteBayesianNetwork([
    ('O', 'H'),  
    ('O', 'W'), 
    ('H', 'R'),  
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C') 
])

print("Independencies in the Bayesian Network:")
print(model.get_independencies())

'''plt.figure(figsize=(10, 8))
pos = nx.spring_layout(model, seed=42)  
nx.draw(model, pos, with_labels=True, node_size=3000, node_color="lightblue", 
        font_size=12, font_weight="bold", arrows=True, arrowsize=20, 
        edge_color="gray", arrowstyle='->')
plt.title("Bayesian Network", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()'''

#b
cpd_o = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]])
cpd_h = TabularCPD(variable='H', variable_card=2, 
                   values=[[0.9, 0.2], [0.1, 0.8]], 
                   evidence=['O'], evidence_card=[2])
cpd_w = TabularCPD(variable='W', variable_card=2, 
                   values=[[0.1, 0.6], [0.9, 0.4]], 
                   evidence=['O'], evidence_card=[2])
cpd_r = TabularCPD(variable='R', variable_card=2, 
                   values=[[0.6, 0.9, 0.3, 0.5], [0.4, 0.1, 0.7, 0.5]], 
                   evidence=['H', 'W'], evidence_card=[2, 2])
cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.8, 0.2], [0.2, 0.8]],
                   evidence=['H'], evidence_card=[2])
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.85, 0.4], [0.15, 0.6]],
                   evidence=['R'], evidence_card=[2])

model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)

assert model.check_model()

inference = VariableElimination(model)

evidence = {'C': 1}
result = inference.query(variables=['H'], evidence=evidence)
print(f"Evidence: {evidence}")
print(result)

result = inference.query(variables=['E'], evidence=evidence)
print(f"Evidence: {evidence}")
print(result)

result = inference.query(variables=['H', 'W'], evidence=evidence)

print("Posterior probability for (H, W) given C=1:")
print(result)