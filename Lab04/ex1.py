from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#a
model = MarkovNetwork()
edges = [("A1", "A2"), ("A1", "A3"), ("A2", "A4"), ("A2", "A5"), ("A3", "A4"), ("A4", "A5")]
model.add_edges_from(edges)
graph = nx.Graph()
graph.add_edges_from(edges)

nx.draw(graph, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
plt.title("Markov Network Graph")
plt.show()

cliques = list(nx.find_cliques(graph))
print("Cliques of the model:", cliques)

#b
factor_A1_A2 = DiscreteFactor(["A1", "A2"], [2, 2], [1, 2, 3, 4])
factor_A1_A3 = DiscreteFactor(["A1", "A3"], [2, 2], [1, 2, 3, 4])
factor_A2_A4 = DiscreteFactor(["A2", "A4"], [2, 2], [1, 2, 3, 4])
factor_A2_A5 = DiscreteFactor(["A2", "A5"], [2, 2], [1, 2, 3, 4])
factor_A3_A4 = DiscreteFactor(["A3", "A4"], [2, 2], [1, 2, 3, 4])
factor_A4_A5 = DiscreteFactor(["A4", "A5"], [2, 2], [1, 2, 3, 4])

model.add_factors(factor_A1_A2, factor_A1_A3, factor_A2_A4, factor_A2_A5, factor_A3_A4, factor_A4_A5)

inference = VariableElimination(model)

joint_distribution = inference.query(variables=["A1", "A2", "A3", "A4", "A5"], joint=True)
print("Joint Distribution:")
print(joint_distribution)

best_configuration_index = np.argmax(joint_distribution.values)
best_configuration = np.unravel_index(best_configuration_index, joint_distribution.cardinality)
print("Best Configuration:")
print(tuple(int(x) for x in best_configuration))
