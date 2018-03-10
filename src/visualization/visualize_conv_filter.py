import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.word_based_cnn import WordBasedCNN
from src.definitions import ROOT_PATH

from scipy.misc import imsave
from keras import backend as K

# Dimensions of the generated picture
img_width = 400
img_height = 30

# Name of the layer we want to visualize
layer_name = "conv1d_1"


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def normalize(x):
    # Normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


model = WordBasedCNN(max_words=30000,
                     max_sequence_length=400,
                     embedding_dim=30,
                     weights_path=os.path.join(
                         ROOT_PATH, "models/cnn_weights.hdf5")).model

layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = layer_dict["embedding_1"].output

num_filters = layer_dict[layer_name].output.shape[2]

filters = []
for filter_index in range(num_filters):
    start_time = time.time()

    # We build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, filter_index])

    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)

    iterate = K.function([input_img], [loss, grads])

    step = 0.3

    input_img_data = np.random.random((1, img_width, img_height))
    input_img_data = (input_img_data - 0.5) * 50 + 128

    # Run gradient ascent
    for i in range(30):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        if loss_value <= 0.:
            # Some filters get stuck to 0, we can skip them
            break

    # Decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        filters.append((img, loss_value))

    end_time = time.time()
    print("Filter {} processed in {:.2f}s".format(
        filter_index, end_time - start_time))

# The filters that have the highest loss are assumed to be better-looking.
best_filter = max(filters, key=lambda x: x[1])
image, _ = best_filter

# Show best filter
plt.imshow(image)
plt.show()

# Save result
filter_path = "conv_filter.png"
imsave("conv_filter.png", image)
print("\n--Filter saved to '{}'".format(filter_path))
