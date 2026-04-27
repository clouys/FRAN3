# Forest Region ANnotation 3

Laboratorio de Robótica y Sistemas Embebidos (LRSE)
Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET) - Universidad de Buenos Aires (UBA)

This application is primarily designed for aerial forest image labeling by using SAM3.  
[SAM 3](https://github.com/facebookresearch/sam3) is a unified foundation model for promptable segmentation in images and videos.

## Installation


1. **Create a new Conda environment:**

```bash
conda create -n sam3 python=3.12 pip
conda deactivate
conda activate sam3
```

2. **Clone the repository and install packages:**

```bash
pip install -e /sam3
```

## Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### Basic Usage

```python
cd sam3_app
python app.py
```
After running these commands, a message will appear redirecting you to a local page. There you will have the application working and you will be ready to begin labeling!

## SAM3 folder

This work contains files from the SAM3 project and is therefore licensed under the SAM [License](sam3/LICENSE).

## Results
The following is an example image of how the interface looks like. Note: the tree labeling may not necessarily be right, as it's an example of how labeled trees are shown providing a different colour for each species, and a grey neutral colour for non-classified trees.
This application lets you choose a colour for each species and assigns you automatically one in case you don't need any special colour for the species.
By holding the cursor and making a box, you can create a prompt that will then be fed to the SAM3 model, to recognize different trees in the image. 
On the right bar, you will see the different masks recognized by the model, and you can either lock them (for the model not to change them), edit them with a brush or eraser (to perfect some details), erase them, isolate them (to view a particular mask alone) or see its assigned species.
By right-clicking a mask, an options menu will appear letting you label the tree, erase the mask, or edit it.
On the left menu bar, you will see, on the upper part, the images uploaded if in folder mode. Folders uploaded should have a json file for each image, specifying some spatial metrics needed to create the polygons afterwards, for them to be correctly displayed in an application such as [QGIS](https://qgis.org). Below, a text prompt will show which lets you type what you want SAM3 to recognize. Finally, an export menu will let you choose the exporting folder or export automatically to a certain destination, downloading a GeoPackage file to that location.

<img width="1695" height="978" alt="Screenshot 2026-04-27 at 11 30 41" src="https://github.com/user-attachments/assets/afa22efc-2ce5-4f0a-8c58-f8e2c3b32381" />

