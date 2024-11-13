from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard

from omni.isaac.core import World
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.simulation_context import SimulationContext

from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Vt, UsdLux, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.particle_system_view import ParticleSystemView
from omni.isaac.core.prims import XFormPrim

import random
import yaml
import os

class Config:
    def __init__(self, config_data=None):
        if config_data:
            self.update_from_dict(config_data)

    def update_from_dict(self, config_data):
        for key, value in config_data.items():
            setattr(self, key, value)

class ParticleEnvironment:
    def __init__(self, config):
        self.config = config
        self.physics_dt = 1/200.0
        self.render_dt = 1/120.0
        self._world = World(stage_units_in_meters=1.0, physics_dt=self.physics_dt, rendering_dt=self.render_dt)
        self.stage = get_current_stage()

        simulation_context = SimulationContext.instance()
        simulation_context.get_physics_context().enable_gpu_dynamics(True)
        simulation_context.get_physics_context().set_broadphase_type("GPU")

        # spawn ground scene
        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=1,
            dynamic_friction=1,
            restitution=0,
        )

        light = UsdLux.DistantLight.Define(self.stage, "/World/defaultDistantLight")
        light.CreateIntensityAttr().Set(5000)

        self.particle_system_path = "/World/particleSystem"

    def create_particle_system(self):
        # Create the particle system with parameters from config
        self._particle_system = ParticleSystem(
            prim_path=self.particle_system_path,
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
        )

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
        )
        physicsUtils.add_physics_material_to_prim(self.stage, self.stage.GetPrimAtPath(self.particle_system_path), pbd_material_path)

        # Create particles from a cylinder mesh
        self.create_particles_from_cylinder_mesh()

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
            Gf.Vec3f(0, 0, 0.6)  # Adjust height as needed
        )
        physicsUtils.set_or_add_scale_op(
            cylinder_mesh, 
            Gf.Vec3f(0.5, 0.5, 1)  # Adjust scale as needed
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

    def setup(self) -> None:
        self.create_particle_system()  # Create the particle system

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

    env = ParticleEnvironment(config)
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
