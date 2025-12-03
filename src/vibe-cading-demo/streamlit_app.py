import streamlit as st
import numpy as np
from visualize import show_voxel

# Mapping prompts to fake voxel data
voxel_library = {
    "cube": "demo_outputs/cube.npy",
    "sphere": "demo_outputs/sphere.npy",
    "handle": "demo_outputs/handle.npy"
}

st.title("Vibe-CADing: Text-to-CAD Demo")

prompt = st.selectbox("Choose a shape description", list(voxel_library.keys()))

if st.button("Generate"):
    voxel = np.load(voxel_library[prompt])
    st.write(f"**Prompt:** {prompt}")
    show_voxel(voxel)
