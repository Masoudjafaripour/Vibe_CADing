import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_voxel(voxel):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel, edgecolor="k")
    ax.view_init(30, 200)
    st.pyplot(fig)
