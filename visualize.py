import visualkeras
from tensorflow.keras.models import load_model
import numpy as np
from PIL import ImageFont

model = load_model("models/model31.h5")

try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = None

color_map = {
    'Conv2D': 'lightblue',
    'MaxPooling2D': 'orange',
    'Flatten': 'pink',
    'Dense': 'darkcyan',
    'Dropout': 'purple',
    'InputLayer': 'gold',
    'BatchNormalization': 'lightgreen',
    'Activation': 'yellow',
}

# Generate a much flatter layered view with dimensions and activation functions in the legend
flat_img = visualkeras.layered_view(
    model,
    legend=True,
    show_dimension=True,  # Show dimensions in the legend
    color_map=color_map,
    font=font,
    scale_xy=20,
    scale_z=2,
    draw_volume=True,
    spacing=10  # Add spacing for better clarity
)
flat_img.save("model_visualization_flat.png")

# Generate a 3D layered view with dimensions and activation functions in the legend
layered_img = visualkeras.layered_view(
    model,
    legend=True,
    show_dimension=True,  # Show dimensions in the legend
    color_map=color_map,
    font=font,
    scale_xy=10,
    scale_z=10,
    draw_volume=True,
    spacing=15  # Adjust spacing for 3D view
)
layered_img.save("model_visualization_3D.png")

# Print layer details including activation functions and number of classes
for layer in model.layers:
    input_shape = getattr(layer, 'input_shape', 'N/A')
    output_shape = getattr(layer, 'output_shape', 'N/A')
    activation = getattr(layer, 'activation', None)
    activation_name = activation.__name__ if activation else 'N/A'
    print(f"Layer: {layer.name}, Input shape: {input_shape}, Output shape: {output_shape}, Activation: {activation_name}")

# Highlight the output layer
output_layer = model.layers[-1]
if hasattr(output_layer, 'output_shape'):
    num_classes = output_layer.output_shape[-1]
    print(f"Output Layer: {output_layer.name}, Number of classes to predict: {num_classes}")
