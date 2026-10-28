from dolfinx import default_scalar_type, fem
from dolfinx.fem import Constant, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile
from dolfinx.io import gmsh
from mpi4py import MPI
from ufl import Measure, TestFunctions, TrialFunctions, div, inner, dot, FacetNormal
import numpy as np
import basix.ufl

mesh_data = gmsh.read_from_msh("./mesh.msh", MPI.COMM_WORLD, gdim=2)
msh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, msh.topology.dim)

dx = Measure("dx", domain=msh, subdomain_data=cell_tags)
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)
n = FacetNormal(msh)

L_, T_, R_, B_ = 1, 2, 3, 4

Q = fem.functionspace(msh, ("DG", 0))
k = Function(Q)
k_outer, k_inner = 1.0, 10.0
for tag, val in [(1, k_outer), (2, k_inner)]:
    cells = cell_tags.find(tag)
    k.x.array[Q.dofmap.list[cells].flatten()] = val
k.x.scatter_forward()

kinv = 1.0 / k

gL = Constant(msh, default_scalar_type(0.0))
gR = Constant(msh, default_scalar_type(1.0))
gB = Constant(msh, default_scalar_type(0.5))
uT = Constant(msh, default_scalar_type(0.2))
alpha = Constant(msh, default_scalar_type(5.0))
beta = Constant(msh, default_scalar_type(10.0))

cell = msh.basix_cell()
V_flux = basix.ufl.element("RT", cell, 1)
V_u = basix.ufl.element("DG", cell, 0)
W = fem.functionspace(msh, basix.ufl.mixed_element([V_flux, V_u]))

(q, u) = TrialFunctions(W)
(r, v) = TestFunctions(W)

a = (
    inner(kinv * q, r) * dx
    - u * div(r) * dx
    - div(q) * v * dx
    + (1.0 / alpha) * dot(q, n) * dot(r, n) * ds(B_)
    + (1.0 / beta) * dot(q, n) * dot(r, n) * ds(T_)
)

L = (
    - gL * dot(r, n) * ds(L_)
    - gR * dot(r, n) * ds(R_)
    - gB * dot(r, n) * ds(B_)
    - uT * dot(r, n) * ds(T_)
)

problem = LinearProblem(
    a, L,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
    petsc_options_prefix="mixed_1_4_",
)
w = problem.solve()

q_h = w.sub(0).collapse()
u_h = w.sub(1).collapse()
q_h.name = "flux"
u_h.name = "u"

V_vis = fem.functionspace(msh, ("DG", 1, (msh.geometry.dim,)))
q_vis = Function(V_vis)
q_vis.interpolate(q_h)
q_vis.name = "flux"

with VTKFile(MPI.COMM_WORLD, "result/mixed_1_4_u.pvd", "w") as f:
    f.write_function(u_h)
with VTKFile(MPI.COMM_WORLD, "result/mixed_1_4_q.pvd", "w") as f:
    f.write_function(q_vis)