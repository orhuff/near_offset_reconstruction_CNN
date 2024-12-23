# near_offset_reconstruction_CNN
Code for convolutional neural networks that are trained on synthetics to reconstruct the near offset gap in marine seismic data. Publication describing the work is available here: https://doi.org/10.1111/1365-2478.13505

The "Example_predictions.ipynb" Jupyter Notebook shows near offset reconstruction examples on a synthetic CMP gather, a synthetic shot gather, and a field TopSeis CMP gather. Note that if you want to use the network on your own CMP or shot gathers, they have to be of shape 448 X 40 (time samples X offset samples) for CMPs, or shape 448 X 72 for shots, as these are the shapes the network was trained on.

The "example_data" folder contains a synthetic CMP gather, synthetic shot gather, and field TopSeis CMP gather, as .npy files.

The "models" folder contains the weights as .h5 files to CMP-trained and shot-trained CNNs. The architecture is first loaded using the "unet_architecture.py" file and a "ModelBuilder" class in the "Example_predictions.ipynb" notebook.

Example Devito code for generating synthetics along a 2D line is located in the "synthetic_generation_code" folder. The velocity model isn't available since it is very close to real data, but you can input your own velocity as a 2D line. Note that Devito needs a 'gcc' compiler and may be difficult to get working on Windows. Here is a guide that might help with that: https://medium.com/@soulfoodpst/install-devito-on-windows-a-tool-for-automatic-programming-3def3949e5c8


