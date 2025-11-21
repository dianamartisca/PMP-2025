import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from hmmlearn import hmm

states = ["Walking", "Running", "Resting"]
n_states = len(states)
observations = ["Low", "Medium", "High"]
obs_index = {o: i for i, o in enumerate(observations)}
startprob = np.array([0.4, 0.3, 0.3])
transmat = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5],
])
emissionprob = np.array([
    [0.10, 0.70, 0.20],  # Walking
    [0.05, 0.25, 0.70],  # Running
    [0.80, 0.15, 0.05],  # Resting
])

def build_hmm():
    model = hmm.MultinomialHMM(n_components=n_states, init_params="")
    model.n_features = len(observations)
    model.n_trials = 1
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    return model

def draw_state_diagram(transmat, states, filename=None):
    G = nx.DiGraph()
    for i, s in enumerate(states):
        G.add_node(s)
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            p = transmat[i, j]
            if p > 0:
                G.add_edge(si, sj, weight=p)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=1800, node_color="#ffd99c")
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("State-transition diagram")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def infer_hidden_states(model, obs_sequence):
    n_samples = len(obs_sequence)
    obs_onehot = np.zeros((n_samples, model.n_features), dtype=int)
    for t, o in enumerate(obs_sequence):
        obs_onehot[t, obs_index[o]] = 1
    logprob, state_sequence = model.decode(obs_onehot, algorithm="viterbi")
    return logprob, state_sequence, obs_onehot

def main():
    observed_sequence = ["Medium", "High", "Low"]
    model = build_hmm()
    print("Start probabilities:", model.startprob_)
    print("Transition matrix:\n", model.transmat_)
    print("Emission probabilities:\n", model.emissionprob_)
    draw_state_diagram(model.transmat_, states)
    viterbi_logprob, state_seq, obs_onehot = infer_hidden_states(model, observed_sequence)
    print(f"\nViterbi (log) joint probability: {viterbi_logprob:.6f}")
    viterbi_joint_prob = float(np.exp(viterbi_logprob))
    print(f"Viterbi joint probability: {viterbi_joint_prob:.6e}")
    print("Most likely state sequence (Viterbi):")
    print([states[s] for s in state_seq])

if __name__ == "__main__":
    main()
