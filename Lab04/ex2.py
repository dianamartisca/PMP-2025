import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

image_size = 5
noise_percentage = 0.1
lambda_param = 1.0
cardinality = 4

original_image = np.random.randint(0, cardinality, (image_size, image_size))

noisy_image = original_image.copy()
num_noisy_pixels = round(noise_percentage * image_size * image_size)
noisy_indices = np.random.choice(image_size * image_size, num_noisy_pixels, replace=False)
for idx in noisy_indices:
    x, y = divmod(idx, image_size)
    noisy_image[x, y] = np.random.randint(0, cardinality)

model = MarkovNetwork()

for i in range(image_size):
    for j in range(image_size):
        if i > 0:
            model.add_edge((i, j), (i - 1, j))
        if i < image_size - 1:
            model.add_edge((i, j), (i + 1, j))
        if j > 0:
            model.add_edge((i, j), (i, j - 1))
        if j < image_size - 1:
            model.add_edge((i, j), (i, j + 1))

epsilon = 1e-10

factors = []
for i in range(image_size):
    for j in range(image_size):
        factors.append(DiscreteFactor([(i, j)], [cardinality], [lambda_param * (noisy_image[i, j] - k)**2 + epsilon for k in range(cardinality)]))

        if i > 0:
            factors.append(DiscreteFactor([(i, j), (i - 1, j)], [cardinality, cardinality], [(k1 - k2)**2 + epsilon for k1 in range(cardinality) for k2 in range(cardinality)]))
        if j > 0:
            factors.append(DiscreteFactor([(i, j), (i, j - 1)], [cardinality, cardinality], [(k1 - k2)**2 + epsilon for k1 in range(cardinality) for k2 in range(cardinality)]))

model.add_factors(*factors)

inference = BeliefPropagation(model)

clean_image = np.zeros((image_size, image_size), dtype=int)
for i in range(image_size):
    for j in range(image_size):
        map_result = inference.map_query(variables=[(i, j)])
        clean_image[i, j] = map_result[(i, j)]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Cleaned Image (MAP)")
plt.imshow(clean_image, cmap="gray")
plt.show()