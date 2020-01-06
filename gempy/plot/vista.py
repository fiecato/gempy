"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software
    Foundation, either version 3 of the License, or (at your option) any later
    version.

    gempy is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along with
    gempy.  If not, see <http://www.gnu.org/licenses/>.


    Module with classes and methods to visualized structural geology data and
    potential fields of the regional modelling based on the potential field
    method. Tested on Ubuntu 18, MacOS 12

    Created on 16.10.2019

    @author: Miguel de la Varga, Bane Sullivan, Alexander Schaaf
"""

import os
import matplotlib.colors as mcolors
import copy
import pandas as pn
import numpy as np
import sys
import gempy as gp
import warnings
# insys.path.append("../../pyvista")
from typing import Union, Dict, List, Iterable
import pyvista as pv
from pyvista.plotting.theme import parse_color

warnings.filterwarnings("ignore",
                        message='.*Conversion of the second argument of issubdtype *.',
                        append=True)
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    VTK_IMPORT = True
except ImportError:
    VTK_IMPORT = False

from nptyping import Array
from logging import debug


class Vista:
    def __init__(
        self, 
        model:gp.Model, 
        extent:List[float]=None, 
        color_lot: pn.DataFrame = None, 
        real_time:bool=False,
        plotter_type:Union["basic", "background"]='basic',
        **kwargs
        ):
        """GemPy 3-D visualization using pyVista.
        
        Args:
            model (gp.Model): Geomodel instance with solutions.
            extent (List[float], optional): Custom extent. Defaults to None.
            color_lot (pn.DataFrame, optional): Custom color scheme in the form
                of a look-up table. Defaults to None.
            real_time (bool, optional): Toggles real-time updating of the plot. 
                Defaults to False.
            plotter_type (Union["basic", "background"], optional): Set the 
                plotter type. Defaults to 'basic'.
        """
        self.model = model
        if extent:
            self.extent = list(extent)
        else:
            self.extent = list(model.grid.regular_grid.extent)
        
        if color_lot:
            self._color_lot = color_lot
        else:
            self._color_lot = model.surfaces.df.set_index('surface')['color']
        self._color_id_lot = model.surfaces.df.set_index('id')['color']

        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
        elif plotter_type == 'background':
            self.p = pv.BackgroundPlotter(**kwargs)

        self._surface_actors = {}
        self._surface_points_actors = {}
        self._orientations_actors = {}

        self._actors = []
        self._live_updating = False

        self.topo_edges = None
        self.topo_ctrs = None

    def _actor_exists(self, new_actor):
        if not hasattr(new_actor, "points"):
            return False
        for actor in self._actors:
            actor_str = actor.points.tostring()
            if new_actor.points.tostring() == actor_str:
                debug("Actor already exists.")
                return True
        return False

    def set_bounds(
            self, 
            extent:list=None, 
            grid:bool=False,
            location:str='furthest', 
            **kwargs
        ):
        """Set and toggle display of bounds of geomodel.
        
        Args:
            extent (list, optional): [description]. Defaults to None.
            grid (bool, optional): [description]. Defaults to False.
            location (str, optional): [description]. Defaults to 'furthest'.
        """
        if extent is None:
            extent = self.extent
        self.p.show_bounds(
            bounds=extent, location=location, grid=grid, **kwargs
        )

    def plot_surface_points(self, fmt:str, **kwargs):
        i = self.model.surface_points.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        mesh = pv.PolyData(
                self.model.surface_points.df.loc[i][["X", "Y", "Z"]].values
            )
        if self._actor_exists(mesh):
                return

        self.p.add_mesh(
            mesh,
            color=self._color_lot[fmt],
            **kwargs
        )
        self._actors.append(mesh)

    def plot_orientations(self, fmt:str, length:float=200, **kwargs):
        i = self.model.orientations.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        pts = self.model.orientations.df.loc[i][["X", "Y", "Z"]].values
        nrms = self.model.orientations.df.loc[i][["G_x", "G_y", "G_z"]].values
        for pt, nrm in zip(pts, nrms):
            mesh = pv.Line(pointa=pt, pointb=pt+length*nrm)
            # TODO: make orientation length func of extent
            if self._actor_exists(mesh):
                return
            self.p.add_mesh(
                mesh, 
                color=self._color_lot[fmt],
                **kwargs
            )
            self._actors.append(mesh)

    def plot_surface_points_all(self, **kwargs):
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_surface_points(fmt, **kwargs)

    def plot_orientations_all(self, **kwargs):
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_orientations(fmt, **kwargs)

    def get_surface(self, fmt:str):
        i = np.where(self.model.surfaces.df.surface == fmt)[0][0]
        ver = self.model.solutions.vertices[i]
        
        sim = self._simplices_to_pv_tri_simplices(
            self.model.solutions.edges[i]
        )
        mesh = pv.PolyData(ver, sim)
        return mesh

    def plot_surface(self, fmt:str, clip:Union[bool, list]=False, **kwargs):
        mesh = [self.get_surface(fmt)]
        if self._actor_exists(mesh):
            return

        if clip:
            mesh = self.clip_horizon_with_faults(mesh[0], clip)

        for m in mesh:
            actor = self.p.add_mesh(
                m,
                color=self._color_lot[fmt],
                **kwargs
            )
            self._actors.append(m)
        self._surface_actors[fmt] = mesh
        return mesh

    def clip_horizon_with_faults(
            self,
            horizon:pv.PolyData, 
            faults:Iterable[pv.PolyData], 
            value:float=None
        ) -> List[pv.PolyData]:
        """Clip given horizon surface with given list of fault surfaces. The
        given value represents the distance to clip away from the fault 
        surfaces.
        
        Args:
            horizon (pv.PolyData): The horizon surface to be clipped.
            faults (Iterable[pv.PolyData]): Fault(s) surface(s) to clip with.
            value (float, optional): Set the clipping value of the implicit 
                function (clipping distance from faults). Defaults to 50.
        
        Returns:
            List[pv.PolyData]: Individual clipped horizon surfaces.
        """
        horizons = []
        if not value:
            value = np.mean(self.model.grid.regular_grid.get_dx_dy_dz()[:2])


        horizons.append(
            horizon.clip_surface(faults[0], invert=False, value=value)
        )

        if len(faults) == 1:
            return horizons
        
        for f1, f2 in zip(faults[:-1], faults[1:]):
            horizons.append(
                horizon.clip_surface(
                    f1, invert=True, value=-value
                ).clip_surface(
                    f2, invert=False, value=value
                )
            )
            
        horizons.append(
            horizon.clip_surface(faults[-1], invert=True, value=-value)
        )
        return horizons

    def plot_surfaces_all(self, fmts:Iterable[str]=None, **kwargs):
        """Plot all geomodel surfaces. If given an iterable containing surface
        strings, it will plot all surfaces specified in it.
        
        Args:
            fmts (List[str], optional): Names of surfaces to plot. 
                Defaults to None.
        """
        if not fmts:
            fmts = self.model.surfaces.df.surface[:-1].values
        for fmt in fmts:
            self.plot_surface(fmt, **kwargs)

    @staticmethod
    def _simplices_to_pv_tri_simplices(sim:Array[int, ..., 3]) -> Array[int, ..., 4]:
        """Convert triangle simplices (n, 3) to pyvista-compatible
        simplices (n, 4)."""
        n_edges = np.ones(sim.shape[0]) * 3
        return np.append(n_edges[:, None], sim, axis=1)

    def plot_structured_grid(self, name:str, **kwargs):
        """Plot a structured grid of the geomodel.

        Args:
            name (str): Can be either one of the following
                'lith' - Lithology id block.
                'scalar' - Scalar field block.
                'values' - Values matrix block.
        """
        regular_grid = self.model.grid.regular_grid
        grid_values = regular_grid.values
        grid_3d = grid_values.reshape(*regular_grid.resolution, 3).T
        mesh = pv.StructuredGrid(*grid_3d)

        if name == "lith":
            vals = self.model.solutions.lith_block
            n_faults = self.model.faults.df['isFault'].sum()
            cmap = mcolors.ListedColormap(list(self._color_id_lot[n_faults:]))
            kwargs['cmap'] = kwargs.get('cmap', cmap)
        elif name == "scalar":
            vals = self.model.solutions.scalar_field_matrix.T
        elif name == "values":
            vals = self.model.solutions.values_matrix.T

        mesh.point_arrays[name] = vals
        if self._actor_exists(mesh):
            return
        self._actors.append(mesh)
        self.p.add_mesh(mesh, **kwargs)

    def _callback_surface_points(self, pos, index):
        print(pos, index)
        x, y, z = pos
        self.model.modify_surface_points(
            index, X=x, Y=y, Z=z
        )
        if self._live_updating:
            self._recompute()
            self._update_surface_polydata()

    def _recompute(self, **kwargs):
        gp.compute_model(self.model, compute_mesh=True, **kwargs)
        self.topo_edges, self.topo_ctrs = gp.assets.topology.compute_topology(
            self.model
        )

    def _update_surface_polydata(self):
        surfaces = self.model.surfaces.df
        for surf, (idx, val) in zip(
            surfaces.surface, 
            surfaces[['vertices', 'edges']].dropna().iterrows()
        ):
            polydata = self._surface_actors[surf]
            polydata.points = val["vertices"]
            polydata.faces = np.insert(val['edges'], 0, 3, axis=1).ravel()  

    def plot_surface_points_interactive(self, fmt:str, **kwargs):
        i = self.model.surface_points.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        sphere = self.p.add_sphere_widget(
            self._callback_surface_points,
            center=self.model.surface_points.df.loc[i][["X", "Y", "Z"]].values,
            color=self._color_lot[fmt],
            indices=i,
            phi_resolution=6,
            theta_resolution=6,
            **kwargs
        )
    
    def plot_surface_points_interactive_all(self, **kwargs):
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_surface_points_interactive(fmt, **kwargs)

    def plot_orientations_interactive(self, fmt:str, **kwargs):
        i = self.model.orientations.df.groupby("surface").groups[fmt]
        if len(i) == 0:
            return

        pts = self.model.orientations.df.loc[i][["X", "Y", "Z"]].values
        nrms = self.model.orientations.df.loc[i][["G_x", "G_y", "G_z"]].values

        for index, pt, nrm in zip(i, pts, nrms):
            widget = self.p.add_plane_widget(
                self._callback_orientations,
                normal=nrm,
                origin=pt,
                bounds=self.extent,
                factor=0.15,
                implicit=False,
                pass_widget=True,
                color=self._color_lot[fmt]
            )
            widget.WIDGET_INDEX = index

    def plot_orientations_interactive_all(self, **kwargs):
        for fmt in self.model.surfaces.df.surface:
            if fmt.lower() == "basement":
                continue
            self.plot_orientations_interactive(fmt, **kwargs)

    def _callback_orientations(self, normal, loc, widget):
        i = widget.WIDGET_INDEX
        x, y, z = loc
        gx, gy, gz = normal

        self.model.modify_orientations(
            i, 
            X=x, Y=y, Z=z,
            G_x=gx, G_y=gy, G_z=gz
        )

        if self._live_updating:
            self._recompute()
            self._update_surface_polydata()

    def _scale_topology_centroids(
            self, 
            centroids:Dict[int, Array[float, 3]]
        ) -> Dict[int, Array[float, 3]]:
        """Scale topology centroid coordinates from grid coordinates to 
        physical coordinates.
        
        Args:
            centroids (Dict[int, Array[float, 3]]): Centroid dictionary.
        
        Returns:
            Dict[int, Array[float, 3]]: Rescaled centroid dictionary.
        """
        res = self.model.grid.regular_grid.resolution
        scaling = np.diff(self.extent)[::2] / res

        scaled_centroids = {}
        for n, pos in centroids.items(): 
            pos_scaled = pos * scaling

            # if np.any(np.array(self.extent[:2]) < 0):
            pos_scaled[0] += np.min(self.extent[:2])
            # if np.any(np.array(self.extent[2:4]) < 0):
            pos_scaled[1] += np.min(self.extent[2:4])
            # if np.any(np.array(self.extent[4:]) < 0):
            pos_scaled[2] += np.min(self.extent[4:])
            print(pos_scaled)

            scaled_centroids[n] = pos_scaled

        return scaled_centroids

    def plot_topology_graph(
            self,
            node_kwargs={},
            edge_kwargs={}
        ):
        if self.topo_edges is None or self.topo_ctrs is None:
            print("No topology results stored. Compute geomodel topology first.")
            return

        centroids = self._scale_topology_centroids(self.topo_ctrs)

        for node, pos in centroids.items():
            mesh = pv.PolyData(
                pos
            )
            # TODO: color nodes with lith colors
            # * Requires topo id to lith id lot
            self.p.add_mesh(mesh, **node_kwargs)

        ekwargs = dict(
            color="black",
            line_width=3
        )
        ekwargs.update(edge_kwargs)

        for e1, e2 in self.topo_edges:
            pos1, pos2 = centroids[e1], centroids[e2]
            mesh = pv.Line(
                pointa=pos1,
                pointb=pos2,
            )
            # TODO: color edges as 1/2 lines using lith colors
            # * Requires topo id to lith id lot
            self.p.add_mesh(mesh, **ekwargs)

    def plot_topography(self):
        # TODO: merge topography plotting
        raise NotImplementedError

class _Vista:
    """
    Class to visualize data and results in 3D. Init will create all the render properties while the method render
    model will lunch the window. Using set_surface_points, set_orientations and set_surfaces in between can be chosen what
    will be displayed.
    """

    def __init__(self, model, extent=None, lith_c: pn.DataFrame = None, real_time=False,
                 plotter_type='basic', **kwargs):

        # Override default notebook value
        kwargs['notebook'] = kwargs.get('notebook', True)

        self.model = model
        self.extent = model.grid.regular_grid.extent if extent is None else extent
        self._color_lot = model.surfaces.df.set_index('id')['color'] if lith_c is None else lith_c

        self.s_widget = pn.DataFrame(columns=['val'])
        self.p_widget = pn.DataFrame(columns=['val'])
        self.surf_polydata = pn.DataFrame(columns=['val'])

        self.vista_rgrids_mesh = {}
        self.vista_rgrids_actors = {}
        self.vista_topo_actors = {}
        self.vista_surf_actor = {}

        self.real_time =real_time

        if plotter_type == 'basic':
            self.p = pv.Plotter(**kwargs)
        elif plotter_type == 'background':
            self.p = pv.BackgroundPlotter(**kwargs)

        self.set_bounds()
        self.p.view_isometric(negative=False)

    def update_colot_lot(self, lith_c=None):
        if lith_c is None:
            lith_c = self.model.surfaces.df.set_index('id')['color'] if lith_c is None else lith_c
            # Hopefully this removes the colors that exist in surfaces but not in data
            idx_uniq = self.model.surface_points.df['id'].unique()
            # + basement
            idx = np.append(idx_uniq, idx_uniq.max()+1)
            lith_c = lith_c[idx]
        self._color_lot = lith_c

    def set_bounds(self, extent=None, grid=False, location='furthest', **kwargs):
        if extent is None:
            extent = self.extent
        self.p.show_bounds(bounds=extent,  location=location, grid=grid, **kwargs)

    def plot_structured_grid(self, regular_grid=None, data: Union[dict, gp.Solution, str] = 'Default',
                             name='lith',
                             **kwargs):
        # Remove previous actor with the same name:
        try:
            self.p.remove_actor(self.vista_rgrids_actors[name])
        except KeyError:
            pass

        self.update_colot_lot()
        if regular_grid is None:
            regular_grid = self.model.grid.regular_grid

        g_values = regular_grid.values
        g_3D = g_values.reshape(*regular_grid.resolution, 3).T
        rg = pv.StructuredGrid(*g_3D)

        self.plot_scalar_data(rg, data, name)
        if name == 'lith':
            n_faults = self.model.faults.df['isFault'].sum()
            cmap = mcolors.ListedColormap(list(self._color_lot[n_faults:]))

            kwargs['cmap'] = kwargs.get('cmap', cmap)

        self.vista_rgrids_mesh[name] = rg

        actor = self.p.add_mesh(rg,  **kwargs)
        self.vista_rgrids_actors[name] = actor
        return actor

    def plot_scalar_data(self, regular_grid, data: Union[dict, gp.Solution, str] = 'Default', name='lith'):
        """

        Args:
            regular_grid:
            data: dictionary or solution
            name: if data is a gp.Solutions object, name of the grid that you want to plot.

        Returns:

        """
        if data == 'Default':
            data = self.model.solutions

        if isinstance(data, gp.Solution):
            if name == 'lith':
                data = {'lith': data.lith_block}

            elif name == 'scalar':
                data = {name: data.scalar_field_matrix.T}

            elif name == 'values':
                data = {name: data.values_matrix.T}

        if type(data) == dict:
            for key in data:
                regular_grid.point_arrays[key] = data[key]

        return regular_grid

    def call_back_sphere(self, *args):

        new_center = args[0]
        obj = args[-1]
        # Get which sphere we are moving
        index = obj.WIDGET_INDEX
        try:
            self.call_back_sphere_change_df(index, new_center)
            self.call_back_sphere_move_changes(index)

        except KeyError as e:
            print('call_back_sphere error:', e)

        if self.real_time:
            try:
                self.update_surfaces_real_time()

            except AssertionError:
                print('Not enough data to compute the model')

    # region Surface Points
    def call_back_sphere_change_df(self, index, new_center):
        index = np.atleast_1d(index)
        # Modify Pandas DataFrame
        self.model.modify_surface_points(index, X=[new_center[0]], Y=[new_center[1]], Z=[new_center[2]])

    def call_back_sphere_move_changes(self, indices):
        df_changes = self.model.surface_points.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z', 'id']]
        for index, df_row in df_changes.iterrows():
            new_center = df_row[['X', 'Y', 'Z']].values

            # Update renderers
            s1 = self.s_widget.loc[index, 'val']
            r_f = s1.GetRadius() * 2
            s1.PlaceWidget(new_center[0] - r_f, new_center[0] + r_f,
                           new_center[1] - r_f, new_center[1] + r_f,
                           new_center[2] - r_f, new_center[2] + r_f)

            s1.GetSphereProperty().SetColor(mcolors.hex2color(self._color_lot[df_row['id']]))
                #self.geo_model.surfaces.df.set_index('id')['color'][df_row['id']]))#self.C_LOT[df_row['id']])

    def call_back_sphere_delete_point(self, ind_i):
        """
        Warning this deletion system will lead to memory leaks until the vtk object is reseted (hopefully). This is
        mainly a vtk problem to delete objects
        """
        ind_i = np.atleast_1d(ind_i)
        # Deactivating widget
        for i in ind_i:
            self.s_widget.loc[i, 'val'].Off()

        self.s_widget.drop(ind_i)

    # endregion

    def plot_data(self, surface_points=None, orientations=None, **kwargs):
        # TODO: When we call this method we need to set the plotter.notebook to False!
        self.plot_surface_points(surface_points, **kwargs)
        self.plot_orientations(orientations, **kwargs)

    def plot_surface_points(self, surface_points=None, radio=None, clear=True, **kwargs):
        self.update_colot_lot()
        if clear is True:
            self.p.clear_sphere_widgets()

        # Calculate default surface_points radio
        if radio is None:
            _e = self.extent
            _e_dx = _e[1] - _e[0]
            _e_dy = _e[3] - _e[2]
            _e_dz = _e[5] - _e[4]
            _e_d_avrg = (_e_dx + _e_dy + _e_dz) / 3
            radio = _e_d_avrg * .01

        if surface_points is None:
            surface_points = self.model.surface_points.df

        test_callback = True if self.real_time is True else False

        # This is Bane way. It gives me some error with index slicing
        centers = surface_points[['X', 'Y', 'Z']]
        colors = self._color_lot[surface_points['id']].values
        s = self.p.add_sphere_widget(self.call_back_sphere,
                                     center=centers, color=colors, pass_widget=True,
                                     test_callback=test_callback,
                                     indices=surface_points.index.values,
                                     radius=radio, **kwargs)

        self.s_widget.append(pn.DataFrame(data=s, index=surface_points.index, columns=['val']))

        return self.s_widget

    def call_back_plane(self, normal, origin, obj):
        """
              Function that rules what happens when we move a plane. At the moment we update the other 3 renderers and
              update the pandas data frame
              """
        # Get new position of the plane and GxGyGz
        new_center = origin
        new_normal = normal

        # Get which plane we are moving
        index = obj.WIDGET_INDEX

        self.call_back_plane_change_df(index, new_center, new_normal)
        # TODO: rethink why I am calling this. Technically this happens outside. It is for sanity check?
        self.call_back_plane_move_changes(index)

        if self.real_time:

            try:
                self.update_surfaces_real_time()

            except AssertionError:
                print('Not enough data to compute the model')

        return True

    def call_back_plane_change_df(self, index, new_center, new_normal):
        # Modify Pandas DataFrame
        # update the gradient vector components and its location
        self.model.modify_orientations(index, X=new_center[0], Y=new_center[1], Z=new_center[2],
                                       G_x=new_normal[0], G_y=new_normal[1], G_z=new_normal[2])
        return True

    def call_back_plane_move_changes(self, indices):
        df_changes = self.model.orientations.df.loc[np.atleast_1d(indices)][['X', 'Y', 'Z',
                                                                             'G_x', 'G_y', 'G_z', 'id']]
        for index, new_values_df in df_changes.iterrows():
            new_center = new_values_df[['X', 'Y', 'Z']].values
            new_normal = new_values_df[['G_x', 'G_y', 'G_z']].values
            new_source = vtk.vtkPlaneSource()
            new_source.SetCenter(new_center)
            new_source.SetNormal(new_normal)
            new_source.Update()

            plane1 = self.p_widget.loc[index, 'val']
            #  plane1.SetInputData(new_source.GetOutput())
            plane1.SetNormal(new_normal)
            plane1.SetCenter(new_center[0], new_center[1], new_center[2])

            plane1.GetPlaneProperty().SetColor(
                parse_color(self.model.surfaces.df.set_index('id')['color'][new_values_df['id']]))  # self.C_LOT[new_values_df['id']])
            plane1.GetHandleProperty().SetColor(
                parse_color(self.model.surfaces.df.set_index('id')['color'][new_values_df['id']]))
        return True

    def plot_orientations(self, orientations=None, clear=True, **kwargs):
        self.update_colot_lot()
        if clear is True:
            self.p.clear_plane_widgets()
        factor = kwargs.get('factor', 0.1)
        kwargs['test_callback'] = kwargs.get('test_callback', False)

        if orientations is None:
            orientations = self.model.orientations.df
        for e, val in orientations.iterrows():
            c = self._color_lot[val['id']]
            p = self.p.add_plane_widget(self.call_back_plane,
                                        implicit=False, pass_widget=True,
                                        normal=val[['G_x', 'G_y', 'G_z']],
                                        origin=val[['X', 'Y', 'Z']], color=c,
                                        bounds=self.model.grid.regular_grid.extent,
                                        factor=factor, **kwargs)
            p.WIDGET_INDEX = e
            self.p_widget.at[e] = p
        return self.p_widget

    def plot_surfaces(self, surfaces=None, delete_surfaces=True, **kwargs):
        self.update_colot_lot()
        if delete_surfaces is True:
            for actor in self.vista_surf_actor.values():
                self.delete_surface(actor)

        if surfaces is None:
            surfaces = self.model.surfaces.df
        for idx, val in surfaces[['vertices', 'edges', 'color']].dropna().iterrows():

            surf = pv.PolyData(val['vertices'], np.insert(val['edges'], 0, 3, axis=1).ravel())
            self.surf_polydata.at[idx] = surf
            self.vista_surf_actor[idx] = self.p.add_mesh(surf, parse_color(val['color']), **kwargs)

        self.set_bounds()
        return self.surf_polydata

    def update_surfaces(self):
        surfaces = self.model.surfaces
        # TODO add the option of update specific surfaces
        for idx, val in surfaces.df[['vertices', 'edges', 'color']].dropna().iterrows():
            self.surf_polydata.loc[idx, 'val'].points = val['vertices']
            self.surf_polydata.loc[idx, 'val'].faces = np.insert(val['edges'], 0, 3, axis=1).ravel()

        return True

    def add_surface(self):
        raise NotImplementedError

    def delete_surface(self, actor):
        self.p.remove_actor(actor)
        return True

    def update_surfaces_real_time(self, delete=True):

        try:
            gp.compute_model(self.model, sort_surfaces=False, compute_mesh=True)
        except IndexError:
            print('IndexError: Model not computed. Laking data in some surface')
        except AssertionError:
            print('AssertionError: Model not computed. Laking data in some surface')

        self.update_surfaces()
        return True

    def plot_topography(self, topography = None, scalars='geo_map', **kwargs):
        """

        Args:
            topography: gp Topography object, np.array or None
            scalars:
            **kwargs:

        Returns:

        """
        if topography is None:
            topography = self.model.grid.topography.values
        rgb = False

        # Create vtk object
        cloud = pv.PolyData(topography)

        # Set scalar values
        if scalars == 'geo_map':
            arr_ = np.empty((0, 3), dtype='int')

            # Convert hex colors to rgb
            for val in list(self._color_lot):
                rgb = (255 * np.array(mcolors.hex2color(val)))
                arr_ = np.vstack((arr_, rgb))

            sel = np.round(self.model.solutions.geological_map).astype(int)[0]
          #  print(arr_)
          #  print(sel)

            scalars_val = numpy_to_vtk(arr_[sel-1], array_type=3)
            cm = None
            rgb = True

        elif scalars == 'topography':
            scalars_val = topography[:, 2]
            cm = 'terrain'

        elif type(scalars) is np.ndarray:
            scalars_val = scalars
            scalars = 'custom'
            cm = 'terrain'
        else:
            raise AttributeError()

        topo_actor = self.p.add_mesh(cloud.delaunay_2d(), scalars=scalars_val, cmap=cm, rgb=rgb, **kwargs)
        self.vista_topo_actors[scalars] = topo_actor
        return topo_actor
























