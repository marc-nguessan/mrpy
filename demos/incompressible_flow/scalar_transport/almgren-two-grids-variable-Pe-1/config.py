"""...

"""

from math import *

# Definition of time integration
t_ini = 0.
t_end = 0.50
nt = 200
dt = (t_end - t_ini) / nt

dt_sc = dt / 20

# Definition of the printing options
n_print = min(nt, 600)
#n_print = nt
dt_print = (t_end - t_ini) / n_print

# domain size
L = 1

x_1 = 0.
y_1 = 0.
x_2 = 0.09
y_2 = 0.
x_3 = -0.045
y_3 = 0.045*sqrt(3)
x_4 = -0.045
y_4 = -0.045*sqrt(3)
x_5 = -0.09
y_5 = 0.

F_1 = -150
F_2 = 50
F_3 = 50
F_4 = 50
F_5 = -100
F_6 = 50

# Definition of the domain dimensions
xmin = -L/2.
xmax = L/2.
ymin = -L/2.
ymax = L/2.
zmin = -L/2.
zmax = L/2.

# Definition of the flow characteristics
Re = 100.
nu = 5.e-4 # (m*m)/s
kappa = 5.e-4 # (m*m)/s

# Tree dimmension
dimension = 2

# Trees min_level
min_level = 2
min_level_sc = 2

# Trees max_level
max_level = 7
max_level_sc = 9

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

def s_north(coords, t=0.):

    #return 1.
    return 0.

def s_south(coords, t=0.):

    #return 1.
    return 0.

def s_west(coords, t=0.):

    #return 1.
    return 0.

def s_east(coords, t=0.):

    #return 1.
    return 0.

def s_back(coords, t=0.):

    #return 1.
    return 0.

def s_forth(coords, t=0.):

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
           "s": {"north": ("neumann", s_north),
                 #"north": ("dirichlet", s_north),
                 #"north": ("periodic", s_north),
                 "south": ("neumann", s_south),
                 #"south": ("dirichlet", s_south),
                 #"south": ("periodic", s_south),
                 #"east": ("dirichlet", s_east),
                 #"east": ("periodic", s_east),
                 "east": ("neumann", s_east),
                 #"west": ("dirichlet", s_west),
                 #"west": ("periodic", s_west),
                 "west": ("neumann", s_west),
                 #"back": ("dirichlet", s_back),
                 "back": ("periodic", s_back),
                 #"back": ("neumann", s_back),
                 #"forth": ("dirichlet", s_forth),
                 "forth": ("periodic", s_forth)},
                 #"forth": ("neumann", s_forth)},
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
threshold_parameter = 1.e-3

# Threshold speed propagation
threshold_speed_propagation = 1

# Name of the thresholding operator
#thresholding_operator_module = "thresholding_operators." + "harten_thresholding"
thresholding_operator_module = "mrpy.mr_utils.operators.thresholding." + "predictive_thresholding"

# Name of the scheme class used for the time integration
#class_scheme_name = "mrpy.discretization." + "temporal_impl_expl_euler"
class_scheme_name = "mrpy.discretization." + "temporal_radau2A"

# Name of the scheme used for the time integration of the scalar
scalar_scheme = "mrpy.discretization." + "RK4_scalar"

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

def sc_init(x, y, t=0.):

    return tanh(100*y)

    #if y < 0:
    #    return 0.
    #else:
    #    return 1.

def omega(x, y, t=0.):

    #return 100*(exp(-(1/r_0**2)*((x - x_1)**2 + y**2)) + exp(-(1/r_0**2)*((x - x_2)**2 + y**2)))
    #return 0.5*F_1*(1 + tanh(100*(0.03 - sqrt((x - x_1)**2 + (y - y_1)**2)))) + \
    #       0.5*F_2*(1 + tanh(100*(0.03 - sqrt((x - x_2)**2 + (y - y_2)**2)))) + \
    #       0.5*F_3*(1 + tanh(100*(0.03 - sqrt((x - x_3)**2 + (y - y_3)**2)))) + \
    #       0.5*F_4*(1 + tanh(100*(0.03 - sqrt((x - x_4)**2 + (y - y_4)**2))))
    return 0.5*F_5*(1 + tanh(100*(0.03 - sqrt((x - x_1)**2 + (y - y_1)**2)))) + \
           0.5*F_6*(1 + tanh(100*(0.03 - sqrt((x - x_2)**2 + (y - y_2)**2)))) + \
           0.5*F_6*(1 + tanh(100*(0.03 - sqrt((x - x_5)**2 + (y - y_5)**2))))
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

