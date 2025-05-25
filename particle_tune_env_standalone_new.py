from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard

from isaacsim.core.api import World, SimulationContext
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.policy.examples.robots.anymal import AnymalFlatTerrainPolicy
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.prims import SingleParticleSystem, SingleXFormPrim

from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Vt, UsdLux, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils

import random
import json
import argparse
import yaml
import os
from sim_config import SimConfig


class Config:
    def __init__(self, config_data=None):
        if config_data:
            self.update_from_dict(config_data)

    def update_from_dict(self, config_data):
        for key, value in config_data.items():
            setattr(self, key, value)


class Anymal_runner(object):
    def __init__(self, config, sim_cfg) -> None:
        """
        Creates the simulation world with preset physics_dt and render_dt and creates an Anymal robot inside the warehouse.

        Argument:
        config {Config} -- Configuration object containing parameters.
        """
        self.config = config
        self._sim_config =  SimConfig(sim_cfg)
        physics_dt = self._sim_config.sim_params["dt"]
        render_dt = self._sim_config.sim_params["rendering_dt"]
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt,  rendering_dt=render_dt, backend = "torch", sim_params=self._sim_config.get_physics_params(),)
        self.stage = get_current_stage()

        simulation_context = SimulationContext.instance()
        simulation_context.get_physics_context().enable_gpu_dynamics(True)
        simulation_context.get_physics_context().set_broadphase_type("GPU")

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")



        # Create custom terrain
        self.generate_central_depression_terrain()
        prim_path = "/World/Anymal"
        # self._anymal = AnymalFlatTerrainPolicy(
        #     prim_path="/World/Anymal",
        #     name="Anymal",
        #     usd_path=assets_root_path + "/Isaac/Robots/ANYbotics/anymal_c.usd",
        #     position=np.array([-2.5, 0, 0.7]),
        # )
        # self._sim_config.apply_articulation_settings(
        #     "anymal", get_prim_at_path(prim_path), self._sim_config.parse_actor_config("anymal")
        # )
        self.create_particle_system()  # Create the particle system

        light = UsdLux.DistantLight.Define(self.stage, "/World/defaultDistantLight")
        light.CreateIntensityAttr().Set(5000)

        self._base_command = np.zeros(3)

        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        self.needs_reset = False
        self.first_step = True

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
        # Define paths
        default_prim_path = "/World"
        particle_system_path = default_prim_path + "/particleSystem"

        # Create the particle system
        self._particle_system = SingleParticleSystem(
            prim_path=particle_system_path,
            particle_system_enabled=True,
            simulation_owner="/physicsScene",
            rest_offset=self.config.particle_system_rest_offset,
            contact_offset=self.config.particle_system_contact_offset,
            solid_rest_offset=self.config.particle_system_solid_rest_offset,
            particle_contact_offset=self.config.particle_system_particle_contact_offset,
            max_velocity=self.config.particle_system_max_velocity,
            max_neighborhood=self.config.particle_system_max_neighborhood,
            solver_position_iteration_count=self.config.particle_system_solver_position_iteration_count,
            enable_ccd=self.config.particle_system_enable_ccd,
            # max_depenetration_velocity=self.config.particle_system_max_depenetration_velocity,
        )

        # Create the particle prototype
        self.create_pbd_material()

        # # # Create a grid of particles
        # self.create_particle_grid()

        # Create particles from a mesh
        self.create_particles_from_mesh()

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

        num_samples_x = int(scale_x / particle_spacing) 
        num_samples_y = int(scale_y / particle_spacing) 
        num_samples_z = int(scale_z / particle_spacing) 

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
        print("Number of particles in grid:", len(positions))

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

    def create_particles_from_mesh(self):
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
                self.config.particle_scale_x, 
                self.config.particle_scale_y, 
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



    def create_pbd_material(self):
        ps = PhysxSchema.PhysxParticleSystem.Get(self.stage, Sdf.Path("/World/particleSystem"))
        # Setting up a material density, will be used by both ref & cand because of shared particle system
        pbd_material_path = Sdf.Path("/World/pbdmaterial")
        particleUtils.add_pbd_particle_material(
            self.stage,
            pbd_material_path,
            friction=self.config.pbd_material_friction,
            particle_friction_scale=self.config.pbd_material_particle_friction_scale,
            adhesion=self.config.pbd_material_adhesion,
            particle_adhesion_scale=self.config.pbd_material_particle_adhesion_scale,
            adhesion_offset_scale=self.config.pbd_material_adhesion_offset_scale,
            density=self.config.pbd_material_density,
            damping = self.config.pbd_material_damping,
        )
        physicsUtils.add_physics_material_to_prim(self.stage, ps.GetPrim(), pbd_material_path)

    def setup(self) -> None:
        """
        Set up keyboard listener and add physics callback
        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._world.add_physics_callback("anymal_advance", callback_fn=self.on_physics_step)

    def on_physics_step(self, step_size) -> None:
        """
        Physics call back, initialize robot (first frame) and call robot advance function to compute and apply joint torque
        """
        if self.first_step:
            # self._anymal.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        # else:
        #     self._anymal.forward(step_size, self._base_command)

    def run(self) -> None:
        """
        Step simulation based on rendering downtime
        """
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        Keyboard subscriber callback to when kit is updated.
        """
        # When a key is pressed or released, the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # On pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # On release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


def main():

    # Load the configuration
    config = Config()

    # Load the YAML configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, 'config.yaml')
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    config = Config(config_data=config_data.get("config"))
    sim_cfg = config_data.get("sim_config")
    print("sim_cfg:", sim_cfg)
    runner = Anymal_runner(config=config, sim_cfg=sim_cfg)
    simulation_app.update()
    runner._world.reset()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
