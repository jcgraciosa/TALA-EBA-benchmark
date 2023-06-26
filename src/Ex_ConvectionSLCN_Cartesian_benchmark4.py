# %% [markdown]
# # Constant viscosity convection, Cartesian domain (benchmark)
# 
# 
# 
# This example solves 2D dimensionless isoviscous thermal convection with a Rayleigh number, for comparison with the [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
# 
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
# 

# %%
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import os 
import numpy as np
import sympy
from copy import deepcopy 

from underworld3.utilities import generateXdmf
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # solve locking issue when reading file
#os.environ["HDF5"]
comm = MPI.COMM_WORLD

# %% [markdown]
# ### Set parameters to use 

# %%
Ra = 1e4 #### Rayleigh number

k = 1.0     #### diffusivity

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

tol = 1e-5
res = 80                        ### x and y res of box
nsteps = 1000                   ### maximum number of time steps to run the first model 
epsilon_lr = 1e-9              ### criteria for early stopping; relative change of the Vrms in between iterations  

##########
# parameters needed for saving checkpoints
# can set outdir to None if you don't want to save anything
outdir = "./Bla_test4" 
outfile = outdir + "/conv4_run13_" + str(res)
save_every = 5
#

#prev_res = None
#infile = None
prev_res = 80 # if infile is not None, then this should be set to the previous model resolution
infile = outdir + "/conv4_run12_" + str(prev_res)    # set infile to a value if there's a checkpoint from a previous run that you want to start from

# example infile settings: 
# infile = outfile # will read outfile, but this file will be overwritten at the end of this run 
# infile = outdir + "/convection_16" # file is that of 16 x 16 mesh   


if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok = True)


def saveData(step, outputPath): # from AdvDiff_Cartesian_benchmark-scaled
    
    ### save mesh vars
    fname = f"{outputPath}/mesh_{'step_'}{step:02d}.h5"
    xfname = f"{outputPath}/mesh_{'step_'}{step:02d}.xmf"
    viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE,  comm=PETSc.COMM_WORLD)

    viewer(meshbox.dm)

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    viewer(v_soln._gvec)         # add velocity
    viewer(p_soln._gvec)         # add pressure
    viewer(t_soln._gvec)           # add temperature
    #viewer(density_proj._gvec)   # add density
    # viewer(materialField._gvec)    # add material projection
    #viewer(timeField._gvec)        # add time
    viewer.destroy()              
    generateXdmf(fname, xfname)

# %% [markdown]
# ### Create mesh and variables

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords=(0.0, 0.0), 
                                                maxCoords=(boxLength, boxHeight), 
                                                cellSize=1.0 /res,
                                                qdegree = 3
                                        )

# meshbox = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight), elementRes=(res,res), qdegree = 3)


# %%
# visualise the mesh if in a notebook / serial

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2) # degree = 2
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1) # degree = 1
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3) # degree = 3
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=3) # degree = 3

# additional variable for the gradient
dTdZ = uw.discretisation.MeshVariable(r"\partial T/ \partial \Z", # FIXME: Z should not be a function of x, y, z 
                                      meshbox, 
                                      1, 
                                      degree = 3) # degree = 3

# variable containing stress in the z direction
sigma_zz = uw.discretisation.MeshVariable(r"\sigma_{zz}",  
                                        meshbox, 
                                        1, degree=2) # degree = 3 

x, z = meshbox.X

# projection object to calculate the gradient along Z
dTdZ_calc = uw.systems.Projection(meshbox, dTdZ)
dTdZ_calc.uw_function = t_soln.sym.diff(z)[0]
dTdZ_calc.smoothing = 1.0e-3
dTdZ_calc.petsc_options.delValue("ksp_monitor")


# %% [markdown]
# ### System set-up 
# Create solvers and set boundary conditions

# %%
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# try these
#stokes.petsc_options['pc_type'] = 'lu' # lu if linear
# stokes.petsc_options["snes_max_it"] = 1000
#stokes.petsc_options["snes_type"] = "ksponly"
stokes.tolerance = tol
#stokes.petsc_options["snes_max_it"] = 1000

# stokes.petsc_options["snes_atol"] = 1e-6
# stokes.petsc_options["snes_rtol"] = 1e-6


#stokes.petsc_options["ksp_rtol"]  = 1e-5 # reduce tolerance to increase speed

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

#stokes.petsc_options["snes_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-2
# stokes.petsc_options.delValue("pc_use_amat")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity=viscosity
stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))


#### buoyancy_force = rho0 * (1 + (beta * deltaP) - (alpha * deltaT)) * gravity
# buoyancy_force = (1 * (1. - (1 * (t_soln.sym[0] - tempMin)))) * -1
buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %%
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

adv_diff.theta = 0.5

# Dirichlet boundary conditions for temperature
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5

# %% [markdown]
# ### Set initial temperature field 
# 
# The initial temperature field is set to a sinusoidal perturbation. 

# %%
import math, sympy

if infile is None:
    pertStrength = 0.1
    deltaTemp = tempMax - tempMin

    with meshbox.access(t_soln, t_0):
        t_soln.data[:] = 0.
        t_0.data[:] = 0.

    with meshbox.access(t_soln):
        for index, coord in enumerate(t_soln.coords):
            # print(index, coord)
            pertCoeff = math.cos( math.pi * coord[0]/boxLength ) * math.sin( math.pi * coord[1]/boxLength )
        
            t_soln.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
            t_soln.data[index] = max(tempMin, min(tempMax, t_soln.data[index]))
            
        
    with meshbox.access(t_soln, t_0):
        t_0.data[:,0] = t_soln.data[:,0]

    #meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln, dTdZ, sigma_zz], index=0)

    #saveData(0, outdir) # from AdvDiff_Cartesian_benchmark-scaled

else:
    meshbox_prev = uw.meshing.UnstructuredSimplexBox(
                                                            minCoords=(0.0, 0.0), 
                                                            maxCoords=(boxLength, boxHeight), 
                                                            cellSize=1.0/prev_res,
                                                            qdegree = 3,
                                                            regular = False
                                                        )
    
    # T should have high degree for it to converge
    # this should have a different name to have no errors
    v_soln_prev = uw.discretisation.MeshVariable("U2", meshbox_prev, meshbox_prev.dim, degree=2) # degree = 2
    p_soln_prev = uw.discretisation.MeshVariable("P2", meshbox_prev, 1, degree=1) # degree = 1
    t_soln_prev = uw.discretisation.MeshVariable("T2", meshbox_prev, 1, degree=3) # degree = 3

    # force to run in serial?
    
    v_soln_prev.read_from_vertex_checkpoint(infile + ".U.0.h5", data_name="U")
    p_soln_prev.read_from_vertex_checkpoint(infile + ".P.0.h5", data_name="P")
    t_soln_prev.read_from_vertex_checkpoint(infile + ".T.0.h5", data_name="T")

    #comm.Barrier()
    # this will not work in parallel?
    #v_soln_prev.load_from_h5_plex_vector(infile + '.U.0.h5')
    #p_soln_prev.load_from_h5_plex_vector(infile + '.P.0.h5')
    #t_soln_prev.load_from_h5_plex_vector(infile + '.T.0.h5')

    with meshbox.access(v_soln, t_soln, p_soln):    
        t_soln.data[:, 0] = uw.function.evaluate(t_soln_prev.sym[0], t_soln.coords)
        p_soln.data[:, 0] = uw.function.evaluate(p_soln_prev.sym[0], p_soln.coords)

        #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        v_coords = deepcopy(v_soln.coords)

        v_soln.data[:] = uw.function.evaluate(v_soln_prev.fn, v_coords)

    meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)

    del meshbox_prev
    del v_soln_prev
    del p_soln_prev
    del t_soln_prev


# %% [markdown]
# ### Some plotting and analysis tools 

# %%
# check the mesh if in a notebook / serial
# allows you to visualise the mesh and the mesh variable
'''FIXME: change this so it's better'''

def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = math.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

#print(f'initial v_rms = {v_rms()}')

# %% [markdown]
# #### Surface integrals
# Since there is no uw3 function yet to calculate the surface integral, we define one.  \
# The surface integral of a function, $f_i(\mathbf{x})$, is approximated as:  
# 
# \begin{aligned}
# F_i = \int_V f_i(\mathbf{x}) S(\mathbf{x})  dV  
# \end{aligned}
# 
# With $S(\mathbf{x})$ defined as an un-normalized Gaussian function with the maximum at $z = a$  - the surface we want to evaluate the integral in (e.g. z = 1 for surface integral at the top surface):
# 
# \begin{aligned}
# S(\mathbf{x}) = exp \left( \frac{-(z-a)^2}{2\sigma ^2} \right)
# \end{aligned}
# 
# In addition, the full-width at half maximum is set to 1/res so the standard deviation, $\sigma$ is calculated as: 
# 
# \begin{aligned}
# \sigma = \frac{1}{2}\frac{1}{\sqrt{ 2 log 2}}\frac{1}{res} 
# \end{aligned}
# 

# %%
# function for calculating the surface integral 
def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral

''' set-up surface expressions for calculating Nu number '''
# the full width at half maximum is set to 1/res
sdev = 0.5*(1/math.sqrt(2*math.log(2)))*(1/res) 

up_surface_defn_fn = sympy.exp(-((z - 1)**2)/(2*sdev**2)) # at z = 1
lw_surface_defn_fn = sympy.exp(-(z**2)/(2*sdev**2)) # at z = 0

# %% [markdown]
# ### Main simulation loop

# %%
t_step = 0
time = 0.

timeVal =  np.zeros(nsteps)*np.nan      # time values
vrmsVal =  np.zeros(nsteps)*np.nan      # v_rms values 
NuVal =  np.zeros(nsteps)*np.nan        # Nusselt number values

# %%
#### Convection model / update in time


while t_step < nsteps:
    vrmsVal[t_step] = v_rms()
    timeVal[t_step] = time

    stokes.solve(zero_init_guess=True) # originally True

    delta_t = 1 * stokes.estimate_dt() # originally 0.5
    adv_diff.solve(timestep=delta_t, zero_init_guess=False) # originally False

    # calculate Nusselt number
    #dTdZ_calc.solve()
    #up_int = surface_integral(meshbox, dTdZ.sym[0], up_surface_defn_fn)
    #lw_int = surface_integral(meshbox, t_soln.sym[0], lw_surface_defn_fn)

    #Nu = -up_int/lw_int

    #NuVal[t_step] = -up_int/lw_int

    # stats then loop
    #tstats = t_soln.stats()

    #if uw.mpi.rank == 0:
    #    print("Timestep {}, dt {}".format(t_step, delta_t), flush = True)
            
    #    print(f't_rms = {t_soln.stats()[6]}, v_rms = {vrmsVal[t_step]}, Nu = {NuVal[t_step]}', flush = True)

    ''' save mesh variables together with mesh '''
    if t_step % save_every == 0 and t_step > 0:
        if uw.mpi.rank == 0:
            print("Timestep {}, dt {}, v_rms {}".format(t_step, delta_t, vrmsVal[t_step]), flush = True)
            print("Saving checkpoint for time step: ", t_step, flush = True)
        meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln], index=0)


    # early stopping criterion
    #if t_step > 1 and abs((NuVal[t_step] - NuVal[t_step - 1])/NuVal[t_step]) < epsilon_lr:
    if t_step > 1 and abs((vrmsVal[t_step] - vrmsVal[t_step - 1])/vrmsVal[t_step - 1]) < epsilon_lr:

        if uw.mpi.rank == 0:
            print("Stopping criterion reached ... ", flush = True)

        break

    t_step += 1
    time   += delta_t

# save final mesh variables in the run 
meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln, dTdZ, sigma_zz], index=0)

