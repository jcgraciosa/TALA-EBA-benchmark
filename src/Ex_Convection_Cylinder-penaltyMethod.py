### Convection in annulus example

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

import os

import sympy




# +
### set reference values
outerRadius   = 6370e3
internalRadius= (6370e3 - 660e3) ### UM - LM transition
innerRadius   = 3480e3
refLength     = (outerRadius - innerRadius) ### thickness of mantle

rI   = innerRadius / refLength
rInt = internalRadius / refLength
rO   = outerRadius / refLength

# +
mu    = 1.
kappa = 1.
alpha = 1.

rho   = 1.

T_cmb = 1.
T_surf= 0.

Ra_number = 1e6

res = 0.075

# +
### FS - free slip top, no slip base
### NS - no slip top and base
boundaryConditions = 'FS'


outputPath = f'./output/FAC-mantleConvection-{boundaryConditions}-res={res}-penaltyMethod/'

if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outputPath):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outputPath)

# +


meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=rO, radiusInternal=rInt, radiusInner=rI, cellSize=res, cellSize_Outer=res, qdegree=3)


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, -5.0]

    meshball.vtk(outputPath+"ignore_meshball.vtk")
    pvmesh = pv.read(outputPath+"ignore_meshball.vtk")

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()

# +
v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
t_0    = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)

timeField    = uw.discretisation.MeshVariable("time", meshball, 1, degree=1)
density_proj = uw.discretisation.MeshVariable("rho", meshball, 1, degree=1)
visc         = uw.discretisation.MeshVariable(r"\eta", meshball, 1, degree=1, continuous=True)

SR         = uw.discretisation.MeshVariable(r"\SR", meshball, 1, degree=1, continuous=True)

meshr        = uw.discretisation.MeshVariable(r"r", meshball, 1, degree=1)
# -


swarm = uw.swarm.Swarm(mesh=meshball)
material = uw.swarm.SwarmVariable("Mat", swarm, 1)
swarm.populate(fill_param=1)


# +
# Create Stokes object
stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

### Add constitutive model
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)

# Set solve options here (or remove default values
stokes.petsc_options.delValue("ksp_monitor")

# +
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusion(
    meshball,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
    verbose=False,
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)

adv_diff.constitutive_model.Parameters.diffusivity = kappa

adv_diff.theta = 0.5
# -

### set up swarm distribution
with swarm.access(material):
    material.data[:] = 0
    material.data[np.sqrt(swarm.data[:,0]**2 + swarm.data[:,1]**2) <= rInt] = 1

T_density = rho * (1. - (alpha * (t_soln.sym[0] - T_surf)))

# +
### create projections of vars onto mesh
nodal_rho_calc = uw.systems.Projection(meshball, density_proj)
nodal_rho_calc.uw_function = T_density
nodal_rho_calc.smoothing = 1.0e-3
nodal_rho_calc.petsc_options.delValue("ksp_monitor")

viscosity_calc = uw.systems.Projection(meshball, visc)
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
viscosity_calc.smoothing = 1.0e-3
viscosity_calc.petsc_options.delValue("ksp_monitor")

SR_calc = uw.systems.Projection(meshball, SR)
SR_calc.uw_function = stokes._Einv2
SR_calc.smoothing = 1.0e-3
SR_calc.petsc_options.delValue("ksp_monitor")

def updateFields(time):
    ### density
    nodal_rho_calc.uw_function = T_density
    nodal_rho_calc.smoothing = 1.0e-3
    nodal_rho_calc.solve(_force_setup=True)
    ### viscosity
    viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    viscosity_calc.smoothing = 1.0e-3
    viscosity_calc.solve(_force_setup=True)
    ### SR
    SR_calc.uw_function = stokes._Einv2
    SR_calc.smoothing = 1.0e-3
    SR_calc.solve(_force_setup=True)
    
    
    ### time
    with meshball.access(timeField):
        timeField.data[:,0] = time
# -

# Some useful coordinate stuff
x, y = meshball.CoordinateSystem.X
ra, th = meshball.CoordinateSystem.xR

# +
# Define initial T boundary conditions via a sympy function

import sympy

abs_r = sympy.sqrt(meshball.rvec.dot(meshball.rvec))
init_t = 0.02 * sympy.sin(15.0 * th) * sympy.sin(np.pi * (ra - rI) / (rO - rI)) + (rO - ra) / (rO - rI)

adv_diff.add_dirichlet_bc(T_cmb, "Lower")
adv_diff.add_dirichlet_bc(T_surf, "Upper")

with meshball.access(t_0, t_soln):
    t_0.data[...] = uw.function.evaluate(init_t, t_0.coords, meshball.N).reshape(-1, 1)
    t_soln.data[...] = t_0.data[...]
    
with meshball.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(sympy.sqrt(x**2 + y**2), meshball.data, meshball.N)  # cf radius_fn which is 0->1
# +
import sympy

radius_fn = sympy.sqrt(meshball.X.dot(meshball.X))  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (radius_fn)
gravity_fn = radius_fn

# +
### set up bouyancy force
buoyancy_force = Ra_number * T_density
stokes.bodyforce = buoyancy_force * -1 * unit_rvec

# Add boundary conditions
if boundaryConditions == 'FS':

    hw = 1000.0 / meshball.get_min_radius()
    
    ### mark the surface nodes
    surface_fn = sympy.exp(-(((meshr.sym[0] - rO) / rO) ** 2) * hw)
    
    ### mark the base nodes
    base_fn = sympy.exp(-(((meshr.sym[0] - rI) / rO) ** 2) * hw)

    free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn
    free_slip_penalty_lower = v_soln.sym.dot(unit_rvec) * unit_rvec * base_fn
    
    ### Buoyancy force RHS plus free slip surface enforcement
    ptf = 1e6
    penalty_terms_upper = ptf * free_slip_penalty_upper
    penalty_terms_lower = ptf * free_slip_penalty_lower
    
    ### Free slip upper
    stokes.bodyforce -= penalty_terms_upper # - penalty_terms_lower
    
    ### No slip lower
    stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Lower.name, (0, 1))
else:  
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# +
# ### set up linear viscosity
viscosity_fn = mu

stokes.constitutive_model.Parameters.viscosity = viscosity_fn

# +
### initial solve
if uw.mpi.size == 1:
    stokes.petsc_options['pc_type']   = 'lu'

stokes.tolerance                  = 1/ptf


# check the stokes solve converges
stokes.solve(zero_init_guess=True)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    pvmesh = pv.read(outputPath+"ignore_meshball.vtk")

    with meshball.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.sym[0], meshball.data)
    pvmesh.point_data["Ts"] = uw.function.evaluate(adv_diff._u_star.sym[0], meshball.data)
    pvmesh.point_data["dT"] = uw.function.evaluate(t_soln.sym[0] - adv_diff._u_star.sym[0], meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=5e-4)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def saveData(step, outputPath, time):
    
    ### update projections first
    updateFields(time)


    meshball.petsc_save_checkpoint(meshVars=[v_soln, p_soln, t_soln, density_proj, visc, SR, timeField], outputPath=outputPath, index=step)
    
    
    swarm.petsc_save_checkpoint(outputPath=outputPath, swarmName='swarm', index=step)

# +
### add in variations in viscosity

### average between the bg visc and yielding visc
UM_visc = mu


### set LM visc
LM_visc = mu



### assign viscosity based on location
viscosity_fn = sympy.Piecewise((LM_visc_lim, sympy.sqrt(x**2+y**2) <= rInt), 
                            (UM_visc_lim, True))

# NL viscoplasitc viscosity
stokes.constitutive_model.Parameters.viscosity = viscosity_fn




# +
stokes.petsc_options["snes_max_it"] = 500

stokes.tolerance                   = 1/ptf
# -

step = 0
time = 0.

# +
# Convection model / update in time

while step < 101:

    if uw.mpi.rank == 0:
        print(f"\n\nTimestep: {step}, time: {time}\n\n")
    
    if step % 5 == 0:
        saveData(step=step, outputPath=outputPath, time = time)
        


    stokes.solve(zero_init_guess=False)
    delta_t = adv_diff.estimate_dt()
    adv_diff.solve(timestep=delta_t)
    
    swarm.advection(v_soln.sym, delta_t=delta_t) 

        
    step += 1
    time += delta_t
# -

