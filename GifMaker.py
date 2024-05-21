# Contains code that generates a gif showing the evolution of the decision boundry

import matplotlib.pyplot as plt
import numpy as np
import imageio
import MLP
import os
import shutil


# Takes a list of weights, generates images for each epoch, packs those images into a gif, removes the temporary images
def make_gif_from_weight_history(nn, dataset, weights_history, output_path = "images/decision_boundry.gif", numFrames=30):
    
    # make temp dir to save gif frames in
    TEMP_DIR = "./images/temp"
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # calculate number of frames to generate
    num_frames = min(len(weights_history), numFrames)
    tot_length = len(weights_history)
    frame_paths = []
    for frame in range(num_frames):
        current_index = int(frame * tot_length/num_frames)
        filepath = "/frame"+str(frame)+".png"
        make_frame(nn, weights_history[current_index], dataset, TEMP_DIR+filepath)
        frame_paths.append(filepath)
    
    # Create the GIF using imageio
    images = []
    project_path = os.path.abspath(os.path.dirname(__file__))
    for file_name in frame_paths:
        file_path = os.path.join(project_path, TEMP_DIR+file_name)
        images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    for _ in range(10):
        images.append(imageio.imread(file_path))
    kargs = { 'duration': 50, 'loop':0 }
    imageio.mimsave(output_path, images, **kargs)

    # remove temp dir and its files
    shutil.rmtree(TEMP_DIR)



# Draw decision boundary from weights
def make_frame(nn, Ws, dataset, filepath):
    plt.clf()
    classA = dataset[dataset['y'] == 1]
    classB = dataset[dataset['y'] == -1]

    # Plot the generated dataset 
    plt.scatter(classA["x1"], classA["x2"], c="green", label="Class 1", zorder=2, marker="o")
    plt.scatter(classB["x1"], classB["x2"], c="blue", label="Class 2", zorder=2, marker="x")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.grid()
    plt.legend()

    # Draw decision boundary
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.array([X1.ravel(), X2.ravel()]).T
    Z = nn.predict(X, Ws)
    Z = Z.reshape(X1.shape)
    plt.contour(X1, X2, Z, levels=[0], colors="red", linewidths=1)
    plt.savefig(filepath)
    #plt.show()