"""...

"""

from math import *

# Definition of time integration
t_ini = 0.
t_end = 10.0
nt = 100
dt = (t_end - t_ini) / nt

# Definition of the printing options
n_print = min(nt, 600)
#n_print = nt
dt_print = (t_end - t_ini) / n_print

# domain size
r_0 = 0.1
L = 100*r_0

# Definition of the domain dimensions
xmin = -L/2.
xmax = L/2.
ymin = -L/2.
ymax = L/2.
zmin = -L/2.
zmax = L/2.

# Definition of the flow characteristics
Re = 100.
nu = 1.e-2 # (m*m)/s

# Tree dimmension
dimension = 2

# Tree min_level
min_level = 2

# Tree max_level
max_level = 6

# Tree stencil graduation
stencil_graduation = 1

# Tree stencil prediction
stencil_prediction = 1

# Frequency of the multiresolution transform
mr_freq = 5

# function
# def function(x, y, t=0.):

#     from math import sin, cos, exp, pi

#     return sin(pi*(x+t))*sin(pi*(y+t))

#def function(x):
#
#    from math import tanh
#
#    return tanh(50.*abs(x-1./2.))

#def function(x, y):
#
#    from math import exp, sqrt
#
#    return exp(-30.*sqrt((x+0.5)**2 + (y-0.5)**2)) +  exp(-30.*sqrt((x-0.5)**2 + (y+0.5)**2))

# Definition of the boundary conditions dictionary bc_dict
# # bc_dict gives the value of the north, south, west and
# # east values of every variables involved in the flow
# # computation

def u_north(coords, t=0.):

    #return (Re*nu) / (xmax - xmin)
    #return 1.
    return 0.

def u_south(coords, t=0.):

    #return 1.
    return 0.

def u_west(coords, t=0.):

    #return 1.
    return 0.

def u_east(coords, t=0.):

    #return 1.
    return 0.

def u_back(coords, t=0.):

    #return 1.
    return 0.

def u_forth(coords, t=0.):

    #return 1.
    return 0.

def v_north(coords, t=0.):

    #return 1.
    return 0.

def v_south(coords, t=0.):

    #return 1.
    return 0.

def v_west(coords, t=0.):

    #return 1.
    return 0.

def v_east(coords, t=0.):

    #return 1.
    return 0.

def v_back(coords, t=0.):

    #return 1.
    return 0.

def v_forth(coords, t=0.):

    #return 1.
    return 0.

def p_north(coords, t=0.):

    #return 1.
    return 0.

def p_south(coords, t=0.):

    #return 1.
    return 0.

def p_west(coords, t=0.):

    #return 1.
    return 0.

def p_east(coords, t=0.):

    #return 1.
    return 0.

def p_back(coords, t=0.):

    #return 1.
    return 0.

def p_forth(coords, t=0.):

    #return 1.
    return 0.

def phi_north(coords, t=0.):

    #return 1.
    return 0.

def phi_south(coords, t=0.):

    #return 1.
    return 0.

def phi_west(coords, t=0.):

    #return 1.
    return 0.

def phi_east(coords, t=0.):

    #return 1.
    return 0.

def phi_back(coords, t=0.):

    #return 1.
    return 0.

def phi_forth(coords, t=0.):

    #return 1.
    return 0.

bc_dict = {"u": {"north": ("neumann", u_north),
                 #"north": ("dirichlet", u_north),
                 #"north": ("periodic", u_north),
                 "south": ("neumann", u_south),
                 #"south": ("dirichlet", u_south),
                 #"south": ("periodic", u_south),
                 "east": ("dirichlet", u_east),
                 #"east": ("periodic", u_east),
                 #"east": ("neumann", u_east),
                 "west": ("dirichlet", u_west),
                 #"west": ("periodic", u_west),
                 #"west": ("neumann", u_west),
                 #"back": ("dirichlet", u_back),
                 "back": ("periodic", u_back),
                 #"back": ("neumann", u_back),
                 #"forth": ("dirichlet", u_forth)},
                 "forth": ("periodic", u_forth)},
                 #"forth": ("neumann", u_forth)},
           "v": {#"north": ("neumann", v_north),
                 "north": ("dirichlet", v_north),
                 #"north": ("periodic", v_north),
                 #"south": ("neumann", v_south),
                 "south": ("dirichlet", v_south),
                 #"south": ("periodic", v_south),
                 #"east": ("dirichlet", v_east),
                 #"east": ("periodic", v_east),
                 "east": ("neumann", v_east),
                 #"west": ("dirichlet", v_west),
                 #"west": ("periodic", v_west),
                 "west": ("neumann", v_west),
                 #"back": ("dirichlet", v_back),
                 "back": ("periodic", v_back),
                 #"back": ("neumann", v_back),
                 #"forth": ("dirichlet", v_forth),
                 "forth": ("periodic", v_forth)},
                 #"forth": ("neumann", v_forth)},
           "phi": {#"north": ("neumann", phi_north),
                 "north": ("dirichlet", phi_north),
                 #"north": ("periodic", phi_north),
                 #"south": ("neumann", phi_south),
                 "south": ("dirichlet", phi_south),
                 #"south": ("periodic", phi_south),
                 "east": ("dirichlet", phi_east),
                 #"east": ("periodic", phi_east),
                 #"east": ("neumann", phi_east),
                 "west": ("dirichlet", phi_west),
                 #"west": ("periodic", phi_west),
                 #"west": ("neumann", phi_west),
                 #"back": ("dirichlet", phi_back),
                 "back": ("periodic", phi_back),
                 #"back": ("neumann", phi_back),
                 #"forth": ("dirichlet", phi_forth),
                 "forth": ("periodic", phi_forth)},
                 #"forth": ("neumann", phi_forth)},
           "p": {"north": ("neumann", p_north),
                 #"north": ("dirichlet", p_north),
                 #"north": ("periodic", p_north),
                 "south": ("neumann", p_south),
                 #"south": ("dirichlet", p_south),
                 #"south": ("periodic", p_south),
                 #"east": ("dirichlet", p_east),
                 #"east": ("periodic", p_east),
                 "east": ("neumann", p_east),
                 #"west": ("dirichlet", p_west),
                 #"west": ("periodic", p_west),
                 "west": ("neumann", p_west),
                 #"back": ("dirichlet", p_back),
                 #"back": ("periodic", p_back),
                 "back": ("neumann", p_back),
                 #"forth": ("dirichlet", p_forth),
                 #"forth": ("periodic", p_forth)}}
                 "forth": ("neumann", p_forth)}}

# Name of the prediction operator module used for a given simulation
prediction_operator_module = "mrpy.mr_utils.operators.prediction." + "centered_polynomial_interpolation"

# Threshold parameter
threshold_parameter = 1.e-2

# Threshold speed propagation
threshold_speed_propagation = 1

# Name of the thresholding operator
#thresholding_operator_module = "thresholding_operators." + "harten_thresholding"
thresholding_operator_module = "mrpy.mr_utils.operators.thresholding." + "predictive_thresholding"

# Name of the scheme class used for the time integration
#class_scheme_name = "mrpy.discretization." + "temporal_impl_expl_euler"
class_scheme_name = "mrpy.discretization." + "temporal_radau2A"

# # Names of the six main spatial operators used for the computation of the
# # simulation: divergence_x, divregence_y, gradient_x, gradient_y, laplacian_x,
# # laplacian_y; the name are divided in two parts:
# # the name of the package where the spatial operators modules are stored and the
# # name of the specific python output module
#gradient_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.gradient"
gradient_module_name = "mrpy.spatial_operators." + "haar.2nd_order_ctr_finite_diff.gradient"

#divergence_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.divergence"
divergence_module_name = "spatial_operators." + "haar.2nd_order_ctr_finite_diff.divergence"

#laplacian_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.laplacian"
#laplacian_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.laplacian-bis"
laplacian_module_name = "mrpy.spatial_operators." + "haar.2nd_order_ctr_finite_diff.laplacian"
#laplacian_module_name = "mrpy.spatial_operators." + "haar.2nd_order_ctr_finite_diff.laplacian-bis"

#mass_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.mass"
mass_module_name = "spatial_operators." + "haar.2nd_order_ctr_finite_diff.mass"

#inverse_mass_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.inverse_mass"
inverse_mass_module_name = "spatial_operators." + "haar.2nd_order_ctr_finite_diff.inverse_mass"

# Name of the output file used to print the solution; it is divided in two parts:
# the name of the package where the input/output modules are stored and the
# name of the specific python output module
#output_module_name = "mrpy.io." + "output-1D-gnuplot"
#output_module_name = "mrpy.io." + "output-tikz"
#output_module_name = "mrpy.io." + "output-2D-gnuplot"
output_module_name = "mrpy.io." + "output-xdmf"

# !!!!!! partie ci-dessous a modifier !!!!!!!!
# Definition of a function that gives the exact value of the x-component of the
# velocity over the domain

# Amplitude of the signal
amp = 1.e+0

def u_exact(x, y, t=0.):

    # return pi*cos(pi*x)*sin(pi*y)*exp(-2*pi*pi*nu*t)
    #return amp*sin(pi*(x+t))*sin(pi*(y+t))
    # return exp(-50*(x**2 + y**2))
    return 0.

# Definition of a function that gives the exact value of the y-component of the
# velocity over the domain

def v_exact(x, y, t=0.):

    # return -pi*sin(pi*x)*cos(pi*y)*exp(-2*pi*pi*nu*t)
    #return amp*cos(pi*(x+t))*cos(pi*(y+t))
    # return -exp(-50*(x**2 + y**2))
    return 0.

# Definition of a function that gives the exact value of the pressure over the
# domain

def p_exact(x, y, t=0.):

    # return (pi*pi)/2.*(sin(pi*x)*sin(pi*x) + sin(pi*y)*sin(pi*y))*exp(-4*pi*pi*nu*t)
    #return amp*sin(pi*(x-y+t))
    return 1.

def omega(x, y, t=0.):

    return exp(-(1/r_0**2)*(x**2 + y**2))
    #return 0.5*(1 + tanh((1/r_0**2)*(0.02 - sqrt(x**2 + y**2))))
    #return 0.

def source_term_function_velocity_x(x, y, t=0.):

    #return pi*(2*amp*nu*pi*sin(pi*(x+t))*sin(pi*(y+t)) + \
    #        amp*cos(pi*(x-y+t)) + amp*sin(pi*(x+y+2*t)) + \
    #        amp*amp*sin(pi*(x+t))*cos(pi*(x+t)))
    return 0.

def source_term_function_velocity_y(x, y, t=0.):

    #return pi*(2*amp*nu*pi*cos(pi*(x+t))*cos(pi*(y+t)) - \
    #        amp*cos(pi*(x-y+t)) - amp*sin(pi*(x+y+2*t)) - \
    #        amp*amp*sin(pi*(y+t))*cos(pi*(y+t)))
    return 0.

