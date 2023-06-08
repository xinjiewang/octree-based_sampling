import sys
sys.path.append("../../src")
from pde import PDELayer


def get_rb2_pde_layer(mean=None, std=None, t_crop=2., z_crop=1., x_crop=2., prandtl=1., rayleigh=1e6, use_continuity=False):
    """Get PDE layer corresponding to the RB2 govening equations.

    Args:
        mean: array of length 4 corresponding to the mean of the 4 physical channels, for normalizng the equations. does not normalize if set to None (default).
        std: array of length 4 corresponding to the std of the 4 physical channels, for normalizing
        the equations. does not normalize if set to None (default).
        t_crop: float, physical temporal span of crop.
        z_crop: float, physical z-width of crop.
        x_crop: float, physical x-width of crop.
    """
    # constants
    P = (rayleigh * prandtl)**(-1/2)
    R = (rayleigh / prandtl)**(-1/2)
    # set up variables and equations
    in_vars = 't, x, z'
    out_vars = 'p, b, u, w'
    nt, nz, nx = 1./t_crop, 1./z_crop, 1./x_crop
    eqn_strs = [
        f'{nt}*dif(b,t)-{P}*(({nx})**2*dif(dif(b,x),x)+({nz})**2*dif(dif(b,z),z))             +(u*{nx}*dif(b,x)+w*{nz}*dif(b,z))',
        f'{nt}*dif(u,t)-{R}*(({nx})**2*dif(dif(u,x),x)+({nz})**2*dif(dif(u,z),z))+dif(p,x)    +(u*{nx}*dif(u,x)+w*{nz}*dif(u,z))',
        f'{nt}*dif(w,t)-{R}*(({nx})**2*dif(dif(w,x),x)+({nz})**2*dif(dif(w,z),z))+dif(p,z)-b  +(u*{nx}*dif(w,x)+w*{nz}*dif(w,z))',
    ]
    # a name/identifier for the equations
    eqn_names = ['transport_eqn_b', 'transport_eqn_u', 'transport_eqn_w']

    if use_continuity:
        eqn_strs.append(f'{nx} * dif(u, x) + {nz} * dif(w, z)')
        eqn_names.append('continuity')

    # normalize equations (optional) via change of variables.
    if (mean is not None) or (std is not None):
        # check the validity of mean and std
        if not ((mean is not None) and (std is not None)):
            raise ValueError('mean and std must either be both None, or both arrays of len 4.')
        if not (hasattr(mean, '__len__') and hasattr(mean, '__len__')):
            raise TypeError(("mean and std must be arrays of len 4. instead they are {} and {}"
                             .format(type(mean), type(std))))
        if not (len(mean) == 4 and len(std) == 4):
            raise ValueError(
                ("mean and std must be arrays of len 4. instead they are of len {} and {}"
                 .format(len(mean), len(std))))
        # substitute variables
        subs_dict = {}
        out_vars_list = [v.strip() for v in out_vars.split(',')]
        for idx, var in enumerate(out_vars_list):
            var_subs = f"{var}*{std[idx]}+{mean[idx]}"
            subs_dict[var] = var_subs
    else:
        subs_dict = None

    # initialize the pde layer
    pde_layer = PDELayer(in_vars=in_vars, out_vars=out_vars)

    for eqn_str, eqn_name in zip(eqn_strs, eqn_names):  # add equations
        pde_layer.add_equation(eqn_str, eqn_name, subs_dict=subs_dict)

    return pde_layer  # NOTE: forward method has not yet been updated.

def get_swe_pde_layer(mean=None, std=None, t_crop=2., y_crop=1., x_crop=2., prandtl=1., rayleigh=1e6, use_continuity=False):
    """Get PDE layer corresponding to the SWE govening equations.

    Args:
        mean: array of length 3 corresponding to the mean of the 3 physical channels, for normalizng the equations. does not normalize if set to None (default).
        std: array of length 3 corresponding to the std of the 3 physical channels, for normalizing
        the equations. does not normalize if set to None (default).
        t_crop: float, physical temporal span of crop.
        y_crop: float, physical y-width of crop.
        x_crop: float, physical x-width of crop.
    """
    # constants
    g = 9.81
    H = 100
    beta = 2E-11
    f_0 = 1E-4 
    # set up variables and equations
    in_vars = 't, x, y'
    out_vars = 'eta, u, v'
    nt, ny, nx = 1./t_crop, 1./y_crop, 1./x_crop
    eqn_strs = [ 
        f'{nt}*dif(eta,t)+{nx}*u*dif(eta,x)+{nx}*(eta+{H})*dif(u,x)+{ny}*v*dif(eta,y)+{ny}*(eta+{H})*dif(v,y)',
        f'{nt}*dif(u,t)-({f_0}+{beta}*y)*v+{g}*{nx}*dif(eta,x)',
        f'{nt}*dif(v,t)-({f_0}+{beta}*y)*u+{g}*{ny}*dif(eta,y)',
    ]
    # a name/identifier for the equations
    eqn_names = ['transport_eqn_eta', 'transport_eqn_u', 'transport_eqn_v']

    if use_continuity:
        eqn_strs.append(f'{nx} * dif(u, x) + {ny} * dif(v, y)')
        eqn_names.append('continuity')

    # normalize equations (optional) via change of variables.
    if (mean is not None) or (std is not None):
        # check the validity of mean and std
        if not ((mean is not None) and (std is not None)):
            raise ValueError('mean and std must either be both None, or both arrays of len 3.')
        if not (hasattr(mean, '__len__') and hasattr(mean, '__len__')):
            raise TypeError(("mean and std must be arrays of len 3. instead they are {} and {}"
                             .format(type(mean), type(std))))
        if not (len(mean) == 3 and len(std) == 3):
            raise ValueError(
                ("mean and std must be arrays of len 3. instead they are of len {} and {}"
                 .format(len(mean), len(std))))
        # substitute variables
        subs_dict = {}
        out_vars_list = [v.strip() for v in out_vars.split(',')]
        for idx, var in enumerate(out_vars_list):
            var_subs = f"{var}*{std[idx]}+{mean[idx]}"
            subs_dict[var] = var_subs
    else:
        subs_dict = None

    # initialize the pde layer
    pde_layer = PDELayer(in_vars=in_vars, out_vars=out_vars)

    for eqn_str, eqn_name in zip(eqn_strs, eqn_names):  # add equations
        pde_layer.add_equation(eqn_str, eqn_name, subs_dict=subs_dict)

    return pde_layer  # NOTE: forward method has not yet been updated.