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

class ParticleEnvironment:
    def __init__(self, config, sim_cfg):
        self.config = config
        self._sim_config =  SimConfig(sim_cfg)
        physics_dt = self._sim_config.sim_params["dt"]
        render_dt = self._sim_config.sim_params["rendering_dt"]
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt,  rendering_dt=render_dt, backend = "torch", sim_params=self._sim_config.get_physics_params(),)
        
        # import carb
        # carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
        self.stage = get_current_stage()

        self._world.get_physics_context().enable_gpu_dynamics(True)
        self._world.get_physics_context().set_broadphase_type("GPU")

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
        self._particle_system = SingleParticleSystem(
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
            friction=self.config.pbd_material_friction,
            particle_friction_scale=self.config.pbd_material_particle_friction_scale,
            adhesion=self.config.pbd_material_adhesion,
            particle_adhesion_scale=self.config.pbd_material_particle_adhesion_scale,
            adhesion_offset_scale=self.config.pbd_material_adhesion_offset_scale,
            density=self.config.pbd_material_density,
        )
        physicsUtils.add_physics_material_to_prim(self.stage, self.stage.GetPrimAtPath(self.particle_system_path), pbd_material_path)

        # Create particles from a cylinder mesh
        self.create_particle_grid()

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

        # positions, velocities = (position, [uniform_particle_velocity] * len(position))
        # widths = [2 * solid_rest_offset * 0.5] * len(position)

        # # Define particle point instancer path
        # particle_point_instancer_path = Sdf.Path(particle_system_path + "/particles")

        # # Add the particle set to the point instancer
        # particleUtils.add_physx_particleset_pointinstancer(
        #     self.stage,
        #     particle_point_instancer_path,
        #     Vt.Vec3fArray(positions),
        #     Vt.Vec3fArray(velocities),
        #     particle_system_path,
        #     self_collision=self.config.particle_grid_self_collision,
        #     fluid=self.config.particle_grid_fluid,
        #     particle_group=self.config.particle_grid_particle_group,
        #     particle_mass=self.config.particle_grid_particle_mass,
        #     density=self.config.particle_grid_density,
        # )

        # # Configure particle prototype
        # particle_prototype_sphere = UsdGeom.Sphere.Get(
        #     self.stage, particle_point_instancer_path.AppendChild("particlePrototype0")
        # )
        # particle_prototype_sphere.CreateRadiusAttr().Set(solid_rest_offset)

        velocities = [uniform_particle_velocity] * len(position)          # list[ Gf.Vec3f ]
        widths     = [solid_rest_offset] * len(position)                  # list[ float ] (diameter)

        # USD Points prim that will hold the particles (no PointInstancer)
        particle_points_path = Sdf.Path(particle_system_path + "/particles")

        # Add the particle set directly as USD Points
        particleUtils.add_physx_particleset_points(
            self.stage,
            particle_points_path,
            position,
            velocities,
            widths,
            particle_system_path,
            self_collision=self.config.particle_grid_self_collision,
            fluid=self.config.particle_grid_fluid,
            particle_group=self.config.particle_grid_particle_group,
            particle_mass=self.config.particle_grid_particle_mass,
            density=self.config.particle_grid_density,
        )

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
    sim_cfg = config_data.get("sim_config")

    env = ParticleEnvironment(config, sim_cfg)
    simulation_app.update()
    env.setup()
    simulation_app.update()
    env._world.reset()
    env._world.reset()
    env._world.reset()
    simulation_app.update()
    env.run()
    simulation_app.close()

if __name__ == "__main__":
    main()
