from noise import pnoise2
import matplotlib.pyplot as plt
import numpy as np


# Set image size
width = 800
height = 600

# Generate Perlin noise values
scale = 0.005  # Adjust this value to control the scale of the noise
octaves = 8  # Adjust this value to control the level of detail in the noise
persistence = 0.5  # (amplitude) Adjust this value to control the roughness of the noise
lacunarity = 2.0  # (frequency) Adjust this value to control the frequency of the noise
seed = 0  # Change this value to generate different patterns

noise_array = np.zeros((height, width))
for y in range(height):
    for x in range(width):
        noise_value = pnoise2(
            x * scale, y * scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            repeatx=width, repeaty=height,
            base=seed
        )
        noise_array[y][x] = noise_value

print(np.min(noise_array))
print(np.max(noise_array))
print(noise_array.dtype)

# Normalize noise values to [0, 1] range
# normalized_array = (noise_array - np.min(noise_array)) / (np.max(noise_array) - np.min(noise_array))
normalized_array = (noise_array + 1) / 2

# Create fog/haze image
image = np.empty((height, width), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        intensity = int(normalized_array[y][x] * 255)
        image[y, x] = intensity

# Display and save the resulting image
plt.imshow(image, cmap='gray')
plt.show()