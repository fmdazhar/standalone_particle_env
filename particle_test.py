from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard

from isaacsim.core.api import World, SimulationContext
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import ParticleSystem, SingleParticleSystem
from isaacsim.core.prims import SingleXFormPrim
from omni.physx.scripts import physicsUtils, particleUtils

import random
import yaml
import os
import pxr
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Vt, UsdLux, UsdShade
from sim_config import SimConfig

class Config:
    def __init__(self, config_data=None):
        if config_data:
            self.update_from_dict(config_data)

    def update_from_dict(self, config_data):
        for key, value in config_data.items():
            setattr(self, key, value)
    def get(self, name, default=None):
        """Mimic dict.get() by returning attribute or default."""
        return getattr(self, name, default)

class ParticleEnvironment:
    def __init__(self, config, sim_cfg):
        self.config = config
        self._sim_config =  SimConfig(sim_cfg)
        physics_dt = self._sim_config.sim_params["dt"]
        render_dt = self._sim_config.sim_params["rendering_dt"]
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt,  rendering_dt=render_dt, backend = "torch", sim_params=self._sim_config.get_physics_params(),)
        self.stage = get_current_stage()
        self._world.get_physics_context().enable_gpu_dynamics(True)
        self._world.get_physics_context().set_broadphase_type("GPU")
        
        self.generate_central_depression_terrain()


        light = UsdLux.DistantLight.Define(self.stage, "/World/defaultDistantLight")
        light.CreateIntensityAttr().Set(5000)

        self.particle_system_path = "/World/particleSystem"

    def generate_flat_terrain(self):
        """
        Generate a terrain with a central negative heightfield depression
        that matches the particle grid's position and size.
        """
        # Terrain parameters from config
        width = self.config.terrain_width        # Number of grid points along the width
        length = self.config.terrain_length         # Number of grid points along the length
        vertical_scale = self.config.terrain_vertical_scale  # Meters per heightfield unit
        horizontal_scale = self.config.terrain_horizontal_scale  # Meters per pixel
        platform_height = self.config.terrain_platform_height    # Height of the surrounding platform in meters
        
        # Particle grid parameters
        x_position = self.config.particle_x_position
        y_position = self.config.particle_y_position
        scale_x = self.config.particle_scale_x 
        scale_y = self.config.particle_scale_y 

        # Set terrain origin to match particle grid center
        terrain_origin_x = x_position - ((width - 1) * horizontal_scale) / 2 
        terrain_origin_y = y_position - ((length - 1) * horizontal_scale) / 2 

        # Convert parameters to discrete units
        platform_height_units = int(platform_height / vertical_scale)

        # Create heightfield array
        height_field_raw = np.full((width, length), platform_height_units, dtype=np.float32)

        # Convert heightfield to triangle mesh
        num_rows, num_cols = height_field_raw.shape
        y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
        x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
        yy, xx = np.meshgrid(y, x)

        # Create vertices
        vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = yy.flatten()
        vertices[:, 2] = height_field_raw.flatten() * vertical_scale

        # Create triangles
        triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
        for i in range(num_rows - 1):
            ind0 = np.arange(0, num_cols - 1) + i * num_cols
            ind1 = ind0 + 1
            ind2 = ind0 + num_cols
            ind3 = ind2 + 1
            start = 2 * i * (num_cols - 1)
            stop = start + 2 * (num_cols - 1)
            triangles[start:stop:2, 0] = ind0
            triangles[start:stop:2, 1] = ind3
            triangles[start:stop:2, 2] = ind1
            triangles[start + 1 : stop : 2, 0] = ind0
            triangles[start + 1 : stop : 2, 1] = ind2
            triangles[start + 1 : stop : 2, 2] = ind3

        # Add terrain to stage
        num_faces = triangles.shape[0]
        terrain_mesh = self.stage.DefinePrim("/World/terrain", "Mesh")
        terrain_mesh.GetAttribute("points").Set(vertices)
        terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
        terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

        # Update terrain position to match particle grid
        terrain = SingleXFormPrim(prim_path="/World/terrain", name="terrain", position=[terrain_origin_x, terrain_origin_y, 0.0])

        # Apply collision properties
        collision_api = UsdPhysics.CollisionAPI.Apply(terrain.prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)
        physx_collision_api.GetRestOffsetAttr().Set(0.00)

    def generate_central_depression_terrain(self):
        """
        Generate a terrain with a central negative heightfield depression
        that matches the particle grid's position and size.
        """
        # Terrain parameters from config
        width = self.config.terrain_width        # Number of grid points along the width
        length = self.config.terrain_length         # Number of grid points along the length
        vertical_scale = self.config.terrain_vertical_scale  # Meters per heightfield unit
        horizontal_scale = self.config.terrain_horizontal_scale  # Meters per pixel
        platform_height = self.config.terrain_platform_height    # Height of the surrounding platform in meters
        depression_depth = -1 * self.config.terrain_depression_depth  # Depth of the depression in meters (negative value)

        # Particle grid parameters
        x_position = self.config.particle_x_position
        y_position = self.config.particle_y_position
        scale_x = self.config.particle_scale_x 
        scale_y = self.config.particle_scale_y 

        # Set terrain origin to match particle grid center
        terrain_origin_x = x_position - ((width - 1) * horizontal_scale) / 2 
        terrain_origin_y = y_position - ((length - 1) * horizontal_scale) / 2 

        # Convert parameters to discrete units
        depression_depth_units = int(depression_depth / vertical_scale)
        platform_height_units = int(platform_height / vertical_scale)

        # Create heightfield array
        height_field_raw = np.zeros((width, length), dtype=np.float32)
        height_field_raw[:, :] = platform_height_units

        # Position depression at the center of the terrain
        center_x = width // 2
        center_y = length // 2
        half_size_x = int((scale_x / 2) / horizontal_scale)
        half_size_y = int((scale_y / 2) / horizontal_scale) 

        depression_start_x = center_x - half_size_x
        depression_end_x = center_x + half_size_x
        depression_start_y = center_y - half_size_y
        depression_end_y = center_y + half_size_y

        # Ensure indices are within bounds
        depression_start_x = max(0, depression_start_x)
        depression_end_x = min(width, depression_end_x)
        depression_start_y = max(0, depression_start_y)
        depression_end_y = min(length, depression_end_y)

        # Set heights for depression
        height_field_raw[depression_start_x:depression_end_x, depression_start_y:depression_end_y] = depression_depth_units

        # Convert heightfield to triangle mesh
        num_rows, num_cols = height_field_raw.shape
        y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
        x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
        yy, xx = np.meshgrid(y, x)

        # Create vertices
        vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
        vertices[:, 0] = xx.flatten()
        vertices[:, 1] = yy.flatten()
        vertices[:, 2] = height_field_raw.flatten() * vertical_scale

        # Create triangles
        triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
        for i in range(num_rows - 1):
            ind0 = np.arange(0, num_cols - 1) + i * num_cols
            ind1 = ind0 + 1
            ind2 = ind0 + num_cols
            ind3 = ind2 + 1
            start = 2 * i * (num_cols - 1)
            stop = start + 2 * (num_cols - 1)
            triangles[start:stop:2, 0] = ind0
            triangles[start:stop:2, 1] = ind3
            triangles[start:stop:2, 2] = ind1
            triangles[start + 1 : stop : 2, 0] = ind0
            triangles[start + 1 : stop : 2, 1] = ind2
            triangles[start + 1 : stop : 2, 2] = ind3

        # Add terrain to stage
        num_faces = triangles.shape[0]
        terrain_mesh = self.stage.DefinePrim("/World/terrain", "Mesh")
        terrain_mesh.GetAttribute("points").Set(vertices)
        terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
        terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

        # Update terrain position to match particle grid
        terrain = SingleXFormPrim(prim_path="/World/terrain", name="terrain", position=[terrain_origin_x, terrain_origin_y, 0.0])

        # Apply collision properties
        collision_api = UsdPhysics.CollisionAPI.Apply(terrain.prim)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.02)
        physx_collision_api.GetRestOffsetAttr().Set(0.00)

    def create_particle_system(self):
        fluid = self.config.particle_grid_fluid
        particle_contact_offset = self.config.particle_system_particle_contact_offset

        if fluid:
            fluid_rest_offset = 0.99 * 0.6 * particle_contact_offset
            rest_offset = 0.99 * particle_contact_offset
            solid_rest_offset = 0.99 * particle_contact_offset
            contact_offset = 1 * particle_contact_offset

            self.config.particle_system_fluid_rest_offset = fluid_rest_offset
            self.config.particle_system_rest_offset = rest_offset
            self.config.particle_system_contact_offset = contact_offset
            self.config.particle_system_solid_rest_offset = solid_rest_offset

        self._particle_system = SingleParticleSystem(
            prim_path=self.particle_system_path,
            particle_system_enabled=True,
            simulation_owner="/physicsScene",
            rest_offset=self.config.get("particle_system_rest_offset", None),
            contact_offset=self.config.get("particle_system_contact_offset", None),
            solid_rest_offset=self.config.get("particle_system_solid_rest_offset", None),
            fluid_rest_offset = self.config.get("particle_system_fluid_rest_offset", None),
            particle_contact_offset=self.config.get("particle_system_particle_contact_offset", None),
            max_velocity=self.config.get("particle_system_max_velocity", None),
            max_neighborhood=self.config.get("particle_system_max_neighborhood", None),
            solver_position_iteration_count=self.config.get("particle_system_solver_position_iteration_count", None),
            enable_ccd=self.config.get("particle_system_enable_ccd", None),
            max_depenetration_velocity=self.config.get("particle_system_max_depenetration_velocity", None),
        )
        self._particle_system_view = ParticleSystem(prim_paths_expr=self.particle_system_path)
        self._world.scene.add(self._particle_system_view)

        # physx_ps = PhysxSchema.PhysxParticleSystem.Define(self.stage, self.particle_system_path)
        # physx_ps.CreateParticleSystemEnabledAttr().Set(True)
        # physx_ps.CreateSimulationOwnerRel().AddTarget(self.stage.GetPrimAtPath("/physicsScene").GetPath())
        # physx_ps.CreateRestOffsetAttr().Set(self.config.particle_system_rest_offset)
        # physx_ps.CreateContactOffsetAttr().Set(self.config.particle_system_contact_offset)
        # physx_ps.CreateSolidRestOffsetAttr().Set(self.config.particle_system_solid_rest_offset)
        # physx_ps.CreateParticleContactOffsetAttr().Set(self.config.particle_system_particle_contact_offset)
        # physx_ps.CreateMaxVelocityAttr().Set(self.config.particle_system_max_velocity)
        # physx_ps.CreateMaxNeighborhoodAttr().Set(self.config.particle_system_max_neighborhood)
        # physx_ps.CreateSolverPositionIterationCountAttr().Set(self.config.particle_system_solver_position_iteration_count)
        # physx_ps.CreateEnableCCDAttr().Set(self.config.particle_system_enable_ccd)

        pbd_material_path = Sdf.Path("/World/pbdmaterial")
        particleUtils.add_pbd_particle_material(
            self.stage,
            pbd_material_path,
            friction=self.config.get("pbd_material_friction", None),
            particle_friction_scale=self.config.get("pbd_material_particle_friction_scale", None),
            damping=self.config.get("pbd_material_damping", None),
            viscosity=self.config.get("pbd_material_viscosity", None),
            vorticity_confinement=self.config.get("pbd_material_vorticity_confinement", None),
            surface_tension=self.config.get("pbd_material_surface_tension", None),
            cohesion=self.config.get("pbd_material_cohesion", None),
            adhesion=self.config.get("pbd_material_adhesion", None),
            particle_adhesion_scale=self.config.get("pbd_material_particle_adhesion_scale", None),
            adhesion_offset_scale=self.config.get("pbd_material_adhesion_offset_scale", None),
            gravity_scale=self.config.get("pbd_material_gravity_scale", None),
            lift=self.config.get("pbd_material_lift", None),
            drag=self.config.get("pbd_material_drag", None),
            density=self.config.get("pbd_material_density", None),
            cfl_coefficient=self.config.get("pbd_material_cfl_coefficient", None)

        )
        physicsUtils.add_physics_material_to_prim(self.stage, self.stage.GetPrimAtPath(self.particle_system_path), pbd_material_path)

        # Create particles from a cylinder mesh
        # self.create_particle_grid()

        # self.create_particles_from_cylinder_mesh()

        self.create_particles_from_cube_mesh()
        
    def create_particles_from_cube_mesh(self):
        """
        Creates particles from the specified mesh.
        
        Args:

        """
        default_prim_path = "/World"
        particle_system_path = default_prim_path + "/particleSystem"
        particle_set_path = default_prim_path + "/particles"
        # create a cube mesh that shall be sampled:
        cube_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(self.stage, "/Cube", True))
        cube_resolution = (
            2  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        )
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform", prim_type="Cube", u_patches=cube_resolution, v_patches=cube_resolution, select_new_prim=False
        )        
        cube_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(cube_mesh_path))

        physicsUtils.setup_transform_as_scale_orient_translate(cube_mesh)

        physicsUtils.set_or_add_translate_op(
        cube_mesh, 
        Gf.Vec3f(
            self.config.particle_x_position, 
            self.config.particle_y_position, 
            self.config.particle_z_position
            )
        )
        physicsUtils.set_or_add_scale_op(
            cube_mesh, 
            Gf.Vec3f(
                self.config.particle_scale_x * 0.95, 
                self.config.particle_scale_y* 0.95, 
                self.config.particle_scale_z
            )
        )
        
        # Calculate sampling distance based on particle system parameters
        solid_rest_offset =  self.config.particle_system_solid_rest_offset
        particle_sampler_distance = 2.5 * solid_rest_offset

        # Apply particle sampling on the mesh
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
        # sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(True)  # Set to True if sampling volume, False for surface

        cube_mesh.CreateVisibilityAttr("invisible")

        # create particle set
        points = UsdGeom.Points.Define(self.stage, particle_set_path)
        points.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(71.0 / 255.0, 125.0 / 255.0, 1.0)]))
        particleUtils.configure_particle_set(points.GetPrim(), particle_system_path, 
        self.config.particle_grid_self_collision, self.config.particle_grid_fluid, 
        self.config.particle_grid_particle_group, self.config.particle_grid_particle_mass, self.config.particle_grid_density)

        # reference the particle set in the sampling api
        sampling_api.CreateParticlesRel().AddTarget(particle_set_path)

    def create_particles_from_cylinder_mesh(self):
        # Define paths
        cylinder_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(self.stage, "/Cylinder", True))

        # Create a cylinder mesh
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform", prim_type="Cylinder", select_new_prim=False
        )
        cylinder_mesh = UsdGeom.Mesh.Get(self.stage, cylinder_mesh_path)

        # Set cylinder position above the ground
        physicsUtils.setup_transform_as_scale_orient_translate(cylinder_mesh)

        physicsUtils.set_or_add_translate_op(
            cylinder_mesh,
            Gf.Vec3f(0, 0, 1.2)  # Adjust height as needed
        )
        physicsUtils.set_or_add_scale_op(
            cylinder_mesh, 
            Gf.Vec3f(1, 1, 2)  # Adjust scale as needed
        )

        # Apply particle sampling on the mesh
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cylinder_mesh.GetPrim())
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(True)  # Fill the volume with particles

        cylinder_mesh.CreateVisibilityAttr("invisible")  # Hide the mesh, show only particles

        # Create particle set and configure
        particle_set_path = "/World/particles"
        points = UsdGeom.Points.Define(self.stage, particle_set_path)
        points.CreateDisplayColorAttr().Set([(0.278, 0.49, 1.0)])  # Example particle color
        particleUtils.configure_particle_set(
            points.GetPrim(), self.particle_system_path, self.config.particle_grid_self_collision, self.config.particle_grid_fluid, 
        self.config.particle_grid_particle_group, self.config.particle_grid_particle_mass, self.config.particle_grid_density
        )

        # Reference the particle set in the sampling API
        sampling_api.CreateParticlesRel().AddTarget(particle_set_path)

    def create_particle_grid(self):
        # Define paths
        default_prim_path = "/World"
        particle_system_path = default_prim_path + "/particleSystem"

        # Define the position and size of the particle grid from config
        x_position = self.config.particle_x_position
        y_position = self.config.particle_y_position
        z_position = self.config.particle_z_position
        scale_x = self.config.particle_scale_x
        scale_y = self.config.particle_scale_y
        scale_z = self.config.particle_scale_z

        lower = Gf.Vec3f(x_position - scale_x * 0.5, y_position - scale_y * 0.5, z_position)

        solid_rest_offset = self.config.particle_system_solid_rest_offset
        particle_spacing = 2.5 * solid_rest_offset

        num_samples_x = int(scale_x / particle_spacing) + 1
        num_samples_y = int(scale_y / particle_spacing) + 1
        num_samples_z = int(scale_z / particle_spacing) + 1

        # Jitter factor from config (as a fraction of particle_spacing)
        jitter_factor = self.config.particle_grid_jitter_factor * particle_spacing

        position = [Gf.Vec3f(0.0)] * num_samples_x * num_samples_y * num_samples_z
        uniform_particle_velocity = Gf.Vec3f(0.0)
        ind = 0
        x = lower[0]
        y = lower[1]
        z = lower[2]
        for i in range(num_samples_x):
            for j in range(num_samples_y):
                for k in range(num_samples_z):
                    jitter_x = random.uniform(-jitter_factor, jitter_factor)
                    jitter_y = random.uniform(-jitter_factor, jitter_factor)
                    jitter_z = random.uniform(-jitter_factor, jitter_factor)

                    # Apply jitter to the position
                    jittered_x = x + jitter_x
                    jittered_y = y + jitter_y
                    jittered_z = z + jitter_z
                    position[ind] = Gf.Vec3f(jittered_x, jittered_y, jittered_z)
                    ind += 1
                    z = z + particle_spacing
                z = lower[2]
                y = y + particle_spacing
            y = lower[1]
            x = x + particle_spacing
        positions, velocities = (position, [uniform_particle_velocity] * len(position))
        widths = [2 * solid_rest_offset * 0.5] * len(position)

        # Define particle point instancer path
        particle_point_instancer_path = Sdf.Path(particle_system_path + "/particles")

        # Add the particle set to the point instancer
        particleUtils.add_physx_particleset_pointinstancer(
            self.stage,
            particle_point_instancer_path,
            Vt.Vec3fArray(positions),
            Vt.Vec3fArray(velocities),
            particle_system_path,
            self_collision=self.config.particle_grid_self_collision,
            fluid=self.config.particle_grid_fluid,
            particle_group=self.config.particle_grid_particle_group,
            particle_mass=self.config.particle_grid_particle_mass,
            density=self.config.particle_grid_density,
        )

        # Configure particle prototype
        particle_prototype_sphere = UsdGeom.Sphere.Get(
            self.stage, particle_point_instancer_path.AppendChild("particlePrototype0")
        )
        particle_prototype_sphere.CreateRadiusAttr().Set(solid_rest_offset)
    
    def create_particle_box_collider(
        self,
        path: Sdf.Path,
        side_length,
        height,
        translate: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
        thickness: float = 0.5,
    ):
        """
        Creates an invisible collider box to catch particles. Opening is in y-up

        Args:
            path:           box path (xform with cube collider children that make up box)
            side_length:    inner side length of box
            height:         height of box
            translate:      location of box, w.r.t it's bottom center
            thickness:      thickness of the box walls
        """
        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.MakeInvisible()
        xform_path = xform.GetPath()
        physicsUtils.set_or_add_translate_op(xform, translate=translate)
        cube_width = side_length + 2.0 * thickness
        offset = side_length * 0.5 + thickness * 0.5
        # front and back (+/- x)
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("top"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, 0,  height * 0.5  ))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, cube_width, thickness))

        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("bottom"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(0, 0,  -height * 0.5 ))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(cube_width, cube_width, thickness))

        # left and right:
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("left"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(offset, 0, 0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, cube_width, height))

        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("right"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        physicsUtils.set_or_add_translate_op(cube, Gf.Vec3f(-offset, 0,0))
        physicsUtils.set_or_add_scale_op(cube, Gf.Vec3f(thickness, cube_width, height ))

        # bottom
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("front"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # half‐thickness up from the base
        physicsUtils.set_or_add_translate_op(
            cube,
            Gf.Vec3f(0, -offset, 0)
        )
        # full width/depth, thin height
        physicsUtils.set_or_add_scale_op(
            cube,
            Gf.Vec3f(cube_width, thickness, height)
        )

        # top
        cube = UsdGeom.Cube.Define(self.stage, xform_path.AppendChild("back"))
        cube.CreateSizeAttr().Set(1.0)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # just below the lid
        physicsUtils.set_or_add_translate_op(
            cube,
            Gf.Vec3f(0, offset,0)
        )
        physicsUtils.set_or_add_scale_op(
            cube,
            Gf.Vec3f(cube_width, thickness, height)
        )

        xform_path_str = str(xform_path)

        paths = [
            xform_path_str + "/front",
            xform_path_str + "/back",
            xform_path_str + "/left",
            xform_path_str + "/right",
            xform_path_str + "/bottom",
            xform_path_str + "/top",
        ]
        glassPath = "/World/Looks/OmniGlass"
        if not self.stage.GetPrimAtPath(glassPath):
            mtl_created = []
            omni.kit.commands.execute(
                "CreateAndBindMdlMaterialFromLibrary",
                mdl_name="OmniGlass.mdl",
                mtl_name="OmniGlass",
                mtl_created_list=mtl_created,
                select_new_prim=False,
            )
            glassPath = mtl_created[0]
        
        for path in paths:
            omni.kit.commands.execute(
                "BindMaterial", prim_path=path, material_path=glassPath
            )

    def setup(self) -> None:
        self.create_particle_system()  # Create the particle system
        # define box collider dimensions
        side = max(self.config.particle_scale_x, self.config.particle_scale_y)
        height = 3  # 20% taller than particle stack
        # center the box around the grid on the ground
        translation = Gf.Vec3f(
            self.config.particle_x_position,
            self.config.particle_y_position,
            0
        )
        # path can be anything under /World
        self.create_particle_box_collider(
            path=Sdf.Path("/World/particle_box"),
            side_length=side,
            height=height,
            translate=translation,
            thickness=0.1  # thin walls
        )

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)

def main():
    # Load configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, 'config.yaml')
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    config = Config(config_data=config_data.get("config"))
    sim_cfg = config_data.get("sim_config")

    env = ParticleEnvironment(config, sim_cfg)
    simulation_app.update()
    env.setup()
    simulation_app.update()
    env._world.reset()
    env._world.reset()
    simulation_app.update()
    env.run()
    simulation_app.close()

if __name__ == "__main__":
    main()
