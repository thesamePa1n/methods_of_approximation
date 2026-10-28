from dolfinx import default_scalar_type, fem
from dolfinx.fem import Constant, Function
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile
from dolfinx.io import gmsh
from mpi4py import MPI
from ufl import Measure, TestFunction, TrialFunction, CellVolume, FacetArea, jump
import numpy as np

mesh_data = gmsh.read_from_msh("./mesh.msh", MPI.COMM_WORLD, gdim=2)
msh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, msh.topology.dim)

dx = Measure("dx", domain=msh, subdomain_data=cell_tags)
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)
dS = Measure("dS", domain=msh, subdomain_data=facet_tags)

L_, T_, R_, B_ = 1, 2, 3, 4

V = fem.functionspace(msh, ("DG", 0))
u = TrialFunction(V)
v = TestFunction(V)

k = Function(V)
for tag, val in [(1, 1.0), (2, 10.0)]:
    cells = cell_tags.find(tag)
    k.x.array[V.dofmap.list[cells].flatten()] = val
k.x.scatter_forward()

gL = Constant(msh, default_scalar_type(0.0))
gR = Constant(msh, default_scalar_type(1.0))
gB = Constant(msh, default_scalar_type(0.5))
uT = Constant(msh, default_scalar_type(0.2))
alpha = Constant(msh, default_scalar_type(5.0))
beta = Constant(msh, default_scalar_type(10.0))

vol = CellVolume(msh)
fa = FacetArea(msh)

d_cell = 2 * vol / (3 * fa)

d_int = d_cell("+") + d_cell("-")
k_harm = 2 * k("+") * k("-") / (k("+") + k("-"))

d_bnd = d_cell

a = (
    k_harm / d_int * jump(u) * jump(v) * dS
    + k / d_bnd * u * v * ds(L_)
    + k / d_bnd * u * v * ds(R_)
    + alpha * u * v * ds(B_)
    + beta * u * v * ds(T_)
)
L = (
    k / d_bnd * gL * v * ds(L_)
    + k / d_bnd * gR * v * ds(R_)
    + alpha * gB * v * ds(B_)
    + beta * uT * v * ds(T_)
)

problem = LinearProblem(
    a, L,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="dg0_1_5_",
)
uh = problem.solve()
uh.name = "u"
uh.x.scatter_forward()

print("min/max:", uh.x.array.min(), uh.x.array.max())

# 8. Сохранение
with VTKFile(msh.comm, "result/dg0_solution.pvd", "w") as vtk:
    vtk.write_mesh(msh)
    vtk.write_function(uh)