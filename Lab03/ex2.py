from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('D', 'B'),  
    ('B', 'R')   
])

cpd_d = TabularCPD(variable='D', variable_card=6, values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]])

# Ball added given die roll
cpd_b = TabularCPD(
    variable='B', variable_card=3,  # 0: Black, 1: Red, 2: Blue
    values=[
        [0, 1, 1, 0, 1, 0],  
        [0, 0, 0, 0, 0, 1], 
        [1, 0, 0, 1, 0, 0],  
    ],
    evidence=['D'],
    evidence_card=[6]
)

# Ball drawn given ball added 
cpd_r = TabularCPD(
    variable='R', variable_card=3,  # col 0: Black added, 1: Red added, 2: Blue added
    values=[
        [3/10, 4/10, 3/10],  #Red
        [4/10, 4/10, 5/10],  #Blue
        [3/10, 2/10, 2/10],  #Black
    ],
    evidence=['B'],
    evidence_card=[3]
)

model.add_cpds(cpd_d, cpd_b, cpd_r)

assert model.check_model()

inference = VariableElimination(model)
result = inference.query(variables=['R'])

print("Probability of drawing a red ball:")
print(result)