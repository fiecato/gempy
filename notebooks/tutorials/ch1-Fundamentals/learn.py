# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("../../..")

# Importing GemPy
# from gempy.core import gempy_api as gp
import gempy as gp


# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


geo_model = gp.create_model('Tutorial_ch1-1_Basics')

data_path= '../..'
# Importing the data from CSV-files and setting extent and resolution
gp.init_data(
    geo_model, [0, 2000., 0, 2000., 0, 2000.], [50, 50, 50],
    path_o = data_path+"/data/input_data/tut_chapter1/simple_fault_model_orientations.csv",
    path_i = data_path+"/data/input_data/tut_chapter1/simple_fault_model_points.csv",
    default_values=True
)
geo_model.surfaces


# gp.get_data(geo_model, 'surface_points').head()
# gp.get_data(geo_model, 'orientations').head()
gp.map_series_to_surfaces(
    geo_model,
    {
        "Fault_Series":'Main_Fault',
        "Strat_Series": ('Sandstone_2','Siltstone', 'Shale', 'Sandstone_1', 'basement')
    }, remove_unused_series=True)

# geo_model.series

geo_model.set_is_fault(['Fault_Series'])
geo_model.faults.faults_relations_df
# gp.plot.plot_data(geo_model, direction='y')
# gp.plot.plot_3D(geo_model)

gp.set_interpolation_data(geo_model,
                          compile_theano=True,
                          theano_optimizer='fast_compile',
                          verbose=[])


gp.get_data(geo_model, 'kriging')
geo_model.additional_data.structure_data

sol = gp.compute_model(geo_model, compute_mesh=True, sort_surfaces=False)
gp.plot.plot_section(geo_model, cell_number=25,
                     direction='y', show_data=True)

ver , sim = gp.get_surfaces(geo_model)
for i in range(len(ver)):
    fig = plt.figure()
    x, y, z = ver[i][:, 0], ver[i][:, 1], ver[i][:, 2]
    X, Y = np.meshgrid(x, y)
    Z = np.array([z])
    Z = np.repeat(Z, Z.shape[1], axis=0)
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

