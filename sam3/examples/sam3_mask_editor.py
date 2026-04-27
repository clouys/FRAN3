import gradio as gr
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import tempfile

# Global state
state = {
    "gdf": None,
    "selected": None,
    "species": None
}


def load_species(file):

    species = file.read().decode().splitlines()
    state["species"] = species

    return gr.Dropdown(choices=species, label="Species")


def load_data(image_file, mask_file):

    img = np.array(Image.open(image_file))

    # save mask file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(mask_file.read())
    tmp.close()

    gdf = gpd.read_file(tmp.name)

    state["gdf"] = gdf

    fig = draw_plot(img, gdf)

    return fig


def draw_plot(image, gdf):

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=image,
            x=0,
            y=0,
            sizex=image.shape[1],
            sizey=image.shape[0],
            sizing="stretch",
            layer="below"
        )
    )

    for idx, row in gdf.iterrows():

        geom = row.geometry

        if geom.geom_type == "Polygon":

            x, y = geom.exterior.xy

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    fill="toself",
                    mode="lines",
                    line=dict(width=2),
                    name=str(idx),
                    hovertext=f"Mask {idx}",
                )
            )

    fig.update_yaxes(autorange="reversed")

    return fig


def select_mask(evt: gr.SelectData):

    state["selected"] = evt.index

    return f"Selected mask: {evt.index}"


def assign_species(species):

    idx = state["selected"]

    if idx is None:
        return "No mask selected"

    state["gdf"].loc[idx, "species"] = species

    return f"Mask {idx} → {species}"


def save_annotations():

    path = "labeled_masks.geojson"

    state["gdf"].to_file(path, driver="GeoJSON")

    return f"Saved to {path}"


with gr.Blocks() as demo:

    gr.Markdown("# SAM Mask Labeling Tool")

    with gr.Row():

        image_input = gr.File(label="Upload Image")
        mask_input = gr.File(label="Upload Masks (GeoJSON/GPKG)")
        species_input = gr.File(label="Upload Species List")

    plot = gr.Plot()

    selected_label = gr.Textbox(label="Selected Mask")

    species_dropdown = gr.Dropdown(label="Species")

    assign_button = gr.Button("Assign Species")

    save_button = gr.Button("Save Annotations")

    image_input.change(
        load_data,
        inputs=[image_input, mask_input],
        outputs=plot
    )

    mask_input.change(
        load_data,
        inputs=[image_input, mask_input],
        outputs=plot
    )

    species_input.change(
        load_species,
        inputs=species_input,
        outputs=species_dropdown
    )

   # plot.select(select_mask, outputs=selected_label)

    assign_button.click(
        assign_species,
        inputs=species_dropdown,
        outputs=selected_label
    )

    save_button.click(
        save_annotations,
        outputs=selected_label
    )

demo.launch()