"""...

"""

# Definition of time integration
t_ini = 0.
t_end = 1.0
nt = 400
dt = (t_end - t_ini) / nt

dt_sc = dt / 10

# Definition of the printing options
n_print = min(nt, 500)
#n_print = nt
dt_print = (t_end - t_ini) / n_print

# domain size
d = 1.e-2 # m
L = 30*d

# Definition of the domain dimensions
xmin = -L/4.
xmax = L/4.
ymin = -L/2.
ymax = L/2.
zmin = -L/2.
zmax = L/2.

# Gravitational constant
g = 9.81 # m/(s*s)

kappa = 6.91*1.e-5 # (m*m)/s

# Reference temperature
T_0 = 298.15 # K

# Reference temperature
P_0 = 1.e+5 # Pa

# Perfect gas constant
R = 8.314 # J/(K*mol)

# Molar masses
M_inj = 0.4003*1.e-2 # kg/mol
M_amb = 2.897*1.e-2 # kg/mol

#Dynamic viscosity
mu_inj = 1.918*1.e-5 # kg/(m*s)
mu_amb = 1.792*1.e-5 # kg/(m*s)

nu = (mu_inj + mu_amb) / 2.
#nu = 0.01

# densities
rho_inj = 0.16148 # kg/(m*m*m)
rho_amb = 1.16864 # kg/(m*m*m)

# injection velocity
u_inj = 0.158 # m/s
#u_inj = 0.000 # m/s
#u_inj = 1.158 # m/s

xi = (R*T_0/P_0)*(1./M_inj - 1./M_amb)
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

# Name of the prediction operator module used for a given simulation
prediction_operator_module = "mrpy.mr_utils.operators.prediction." + "centered_polynomial_interpolation"

# Threshold parameter
threshold_parameter = 5.e-3

# Threshold speed propagation
threshold_speed_propagation = 1

# Name of the thresholding operator
#thresholding_operator_module = "thresholding_operators." + "harten_thresholding"
thresholding_operator_module = "mrpy.mr_utils.operators.thresholding." + "predictive_thresholding"

# Name of the scheme class used for the time integration
class_scheme_name = "mrpy.discretization." + "temporal_impl_expl_euler"
#class_scheme_name = "mrpy.discretization." + "temporal_radau2A"
#class_scheme_name = "mrpy.discretization." + "ESDIRK3_2I_4A_2"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4A_2"
#class_scheme_name = "mrpy.discretization." + "ERK_ESDIRK3_2I_4A_2"
#class_scheme_name = "mrpy.discretization." + "RK4_velocity"
#class_scheme_name = "mrpy.discretization." + "ESDIRK3_2I_4L_2_KC"
#class_scheme_name = "mrpy.discretization." + "ERK_ESDIRK3_2I_4L_2_KC_2003"
#class_scheme_name = "mrpy.discretization." + "ERK_ESDIRK3_2I_4L_2_new"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4L_2_KC_2003"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4L_2_new"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4L_2_so"
#class_scheme_name = "mrpy.discretization." + "ESDIRK3_2I_4L_2_SA"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4L_2_SA"
#class_scheme_name = "mrpy.discretization." + "ERK_ESDIRK3_2I_4L_2_SA"
#class_scheme_name = "mrpy.discretization." + "ESDIRK3_2I_4L_2_WBCK"
#class_scheme_name = "mrpy.discretization." + "ARK_ESDIRK3_2I_4L_2_WBCK"
#class_scheme_name = "mrpy.discretization." + "ERK_ESDIRK3_2I_4L_2_WBCK"
#class_scheme_name = "mrpy.discretization." + "HRK4_velocity"
#class_scheme_name = "mrpy.discretization." + "HRK3_Heun_velocity"

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

#density_module_name = "mrpy.spatial_operators." + "ctr_poly.2nd_order_ctr_finite_diff.density"
density_module_name = "spatial_operators." + "haar.2nd_order_ctr_finite_diff.density"

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

    from math import sin, cos, exp, pi

    # return pi*cos(pi*x)*sin(pi*y)*exp(-2*pi*pi*nu*t)
    #return amp*sin(pi*(x+t))*sin(pi*(y+t))
    # return exp(-50*(x**2 + y**2))
    #return sin(2*pi*x)
    return 0.

# Definition of a function that gives the exact value of the y-component of the
# velocity over the domain

def v_exact(x, y, t=0.):

    from math import sin, cos, exp, pi

    # return -pi*sin(pi*x)*cos(pi*y)*exp(-2*pi*pi*nu*t)
    #return amp*cos(pi*(x+t))*cos(pi*(y+t))
    # return -exp(-50*(x**2 + y**2))
    return 0.

# Definition of a function that gives the exact value of the pressure over the
# domain

def p_exact(x, y, t=0.):

    from math import sin, cos, exp, pi

    # return (pi*pi)/2.*(sin(pi*x)*sin(pi*x) + sin(pi*y)*sin(pi*y))*exp(-4*pi*pi*nu*t)
    #return amp*sin(pi*(x-y+t))
    #return 1.
    return 0

def y_1_init(x, y, t=0.):

    return 0.

def y_2_init(x, y, t=0.):

    return 1.

def rho_init(x, y, t=0.):

    temp = 1./M_inj*y_1_init(x, y, t) + 1./M_amb*y_2_init(x, y, t)

    foo = 1/temp
    bar = P_0/(R*T_0)

    return bar*foo

def source_term_function_velocity_x(x, y, t=0.):

    from math import sin, cos, pi

    #return pi*(2*amp*nu*pi*sin(pi*(x+t))*sin(pi*(y+t)) + \
    #        amp*cos(pi*(x-y+t)) + amp*sin(pi*(x+y+2*t)) + \
    #        amp*amp*sin(pi*(x+t))*cos(pi*(x+t)))
    return 0.

def source_term_function_velocity_y(x, y, t=0.):

    from math import sin, cos, pi

    #return pi*(2*amp*nu*pi*cos(pi*(x+t))*cos(pi*(y+t)) - \
    #        amp*cos(pi*(x-y+t)) - amp*sin(pi*(x+y+2*t)) - \
    #        amp*amp*sin(pi*(y+t))*cos(pi*(y+t)))
    return 0.
    #return -g

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

    #x = coords[0]

    #if (x >= - d/2.) and (x <= d/2.):
    #    return u_inj*(1 - 4*x*x /(d*d))
    #else:
    #    return 0

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

    #return 2*u_inj*d/(3*(xmax-xmin))
    return u_inj*d/(xmax-xmin)
    #return 0.

def v_south(coords, t=0.):

    #return 1.

    x = coords[0]

    #if (x >= - d/2.) and (x <= d/2.):
    #    return u_inj*(1 - 4*x*x /(d*d))
    #else:
    #    return 0

    if (x >= - d/2.) and (x <= d/2.):
        return u_inj
    else:
        return 0
    #return 0.

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

def y_1_north(coords, t=0.):

    #return 1.
    return 0.

def y_1_south(coords, t=0.):

    #return 1.

    x = coords[0]

    if (x >= - d/2.) and (x <= d/2.):
        return 1.
    else:
        return 0

    #return 0.

def y_1_west(coords, t=0.):

    #return 1.
    return 0.

def y_1_east(coords, t=0.):

    #return 1.
    return 0.

def y_1_back(coords, t=0.):

    #return 1.
    return 0.

def y_1_forth(coords, t=0.):

    #return 1.
    return 0.

def y_2_north(coords, t=0.):

    #return 1.
    return 0.

def y_2_south(coords, t=0.):

    #return 1.
    return 0.

def y_2_west(coords, t=0.):

    #return 1.
    return 0.

def y_2_east(coords, t=0.):

    #return 1.
    return 0.

def y_2_back(coords, t=0.):

    #return 1.
    return 0.

def y_2_forth(coords, t=0.):

    #return 1.
    return 0.

bc_dict = {"u": {"north": ("neumann", u_north),
                 #"north": ("dirichlet", u_north),
                 #"north": ("periodic", u_north),
                 #"south": ("neumann", u_south),
                 "south": ("dirichlet", u_south),
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
                 "east": ("dirichlet", v_east),
                 #"east": ("periodic", v_east),
                 #"east": ("neumann", v_east),
                 "west": ("dirichlet", v_west),
                 #"west": ("periodic", v_west),
                 #"west": ("neumann", v_west),
                 #"back": ("dirichlet", v_back),
                 "back": ("periodic", v_back),
                 #"back": ("neumann", v_back),
                 #"forth": ("dirichlet", v_forth),
                 "forth": ("periodic", v_forth)},
                 #"forth": ("neumann", v_forth)},
           "y_1": {"north": ("neumann", y_1_north),
                 #"north": ("dirichlet", y_1_north),
                 #"north": ("periodic", y_1_north),
                 #"south": ("neumann", y_1_south),
                 "south": ("dirichlet", y_1_south),
                 #"south": ("periodic", y_1_south),
                 #"east": ("dirichlet", y_1_east),
                 #"east": ("periodic", y_1_east),
                 "east": ("neumann", y_1_east),
                 #"west": ("dirichlet", y_1_west),
                 #"west": ("periodic", y_1_west),
                 "west": ("neumann", y_1_west),
                 #"back": ("dirichlet", y_1_back),
                 "back": ("periodic", y_1_back),
                 #"back": ("neumann", y_1_back),
                 #"forth": ("dirichlet", y_1_forth),
                 "forth": ("periodic", y_1_forth)},
                 #"forth": ("neumann", y_1_forth)},
           "y_2": {"north": ("neumann", y_2_north),
                 #"north": ("dirichlet", y_2_north),
                 #"north": ("periodic", y_2_north),
                 #"south": ("neumann", y_2_south),
                 "south": ("dirichlet", y_2_south),
                 #"south": ("periodic", y_2_south),
                 #"east": ("dirichlet", y_2_east),
                 #"east": ("periodic", y_2_east),
                 "east": ("neumann", y_2_east),
                 #"west": ("dirichlet", y_2_west),
                 #"west": ("periodic", y_2_west),
                 "west": ("neumann", y_2_west),
                 #"back": ("dirichlet", y_2_back),
                 "back": ("periodic", y_2_back),
                 #"back": ("neumann", y_2_back),
                 #"forth": ("dirichlet", y_2_forth),
                 "forth": ("periodic", y_1_forth)},
                 #"forth": ("neumann", y_1_forth)},
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

