from dolfinx import default_scalar_type, fem
from dolfinx.fem import Constant, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile
from dolfinx.io import gmsh
from mpi4py import MPI
from ufl import Measure, TestFunction, TrialFunction, grad, inner

import numpy as np

mesh_data = gmsh.read_from_msh("./mesh.msh", MPI.COMM_WORLD, gdim=2)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tag = mesh_data.facet_tags

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
dx = Measure("dx", domain=mesh, subdomain_data=cell_tags)

V = fem.functionspace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

c = Constant(mesh, default_scalar_type(1))
k_square = Constant(mesh, default_scalar_type(1))
k_circle = Constant(mesh, default_scalar_type(10))

g_L = Constant(mesh, default_scalar_type(0))
g_R = Constant(mesh, default_scalar_type(1))

alpha = Constant(mesh, default_scalar_type(5))
g_B = Constant(mesh, default_scalar_type(0.5))

beta = Constant(mesh, default_scalar_type(10))
u_T = Constant(mesh, default_scalar_type(0.2))

dofs_L = locate_dofs_topological(V, fdim, facet_tag.find(np.int32(1)))
dofs_R = locate_dofs_topological(V, fdim, facet_tag.find(np.int32(3)))

bc_L = dirichletbc(g_L, dofs_L, V)
bc_R = dirichletbc(g_R, dofs_R, V)
bcs = [bc_L, bc_R]

T = 1.0
num_steps = 50
dt = T / num_steps

u_n = fem.Function(V)
u_n.name = "u"

x_min = mesh.comm.allreduce(np.min(mesh.geometry.x[:, 0]), op=MPI.MIN)
x_max = mesh.comm.allreduce(np.max(mesh.geometry.x[:, 0]), op=MPI.MAX)

u_n.interpolate(lambda x: (x[0] - x_min) / (x_max - x_min))
u_n.x.scatter_forward()

a = (
    (c / dt) * u * v * dx(1)
    + (c / dt) * u * v * dx(2)
    + inner(k_square * grad(u), grad(v)) * dx(1)
    + inner(k_circle * grad(u), grad(v)) * dx(2)
    + alpha * u * v * ds(4)
    + beta * u * v * ds(2)
)

L = (
    (c / dt) * u_n * v * dx(1)
    + (c / dt) * u_n * v * dx(2)
    + alpha * g_B * v * ds(4)
    + beta * u_T * v * ds(2)
)

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    },
    petsc_options_prefix="transient_"
)

with VTKFile(mesh.comm, "./result/solution.pvd", "w") as vtk:
    vtk.write_mesh(mesh)
    vtk.write_function(u_n, 0.0)

    for n in range(1, num_steps + 1):
        t = n * dt

        uh = problem.solve()
        uh.name = "u"
        uh.x.scatter_forward()

        vtk.write_function(uh, t)

        u_n.x.array[:] = uh.x.array
        u_n.x.scatter_forward()