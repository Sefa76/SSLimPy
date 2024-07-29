from copy import copy
from typing import Union

import SSLimPy.interface.config as cfg
from SSLimPy.cosmology import astro, cosmology

def update_cosmo(current_cosmo: cosmology.cosmo_functions,
                   cosmopars: dict,
                   ) -> cosmology.cosmo_functions:
    """This function gets the old cosmology aswell as a new set of
    cosmological parameters. It then seperates EBS parameters from
    nuiscance parameters and heirachically computes the new classes
    needed to compute the power spectrum.
    """

    current_cosmopars = current_cosmo.cosmopars
    current_nuiscance_like = current_cosmo.nuiscance_like

    # seperate nuiscance-like cosmo pars
    nuiscance_like = dict()
    for key in cosmopars:
        if key in cfg.nuiscance_like_params_names:
            nuiscance_like[key]=cosmopars.pop(key)

    # EBS results:
    if cosmopars == cfg.fiducialcosmoparams:
        cosmo = cfg.fiducialcosmo
        if nuiscance_like != cfg.fiducialnuiscancelikeparams:
            cosmo = cosmology.cosmo_functions(cosmopars=cosmopars,
                                                nuiscance_like=nuiscance_like,
                                                cosmology=cosmo,
                                                )
    elif cosmopars == current_cosmopars:
        cosmo = current_cosmo
        if nuiscance_like != current_nuiscance_like:
            cosmo = cosmology.cosmo_functions(cosmopars=cosmopars,
                                                nuiscance_like=nuiscance_like,
                                                cosmology=cosmo,
                                                )
    else:
        cosmo = cosmology.cosmo_functions(cosmopars=cosmopars,
                                            nuiscance_like=nuiscance_like,
                                            )

    return cosmo

updated_cosmo_type = Union[cosmology.cosmo_functions, None]
def update_astro(current_cosmo: cosmology.cosmo_functions,
                  cosmopars: dict,
                  current_astro: astro.astro_functions,
                  astropars: dict,
                  updated_cosmo: updated_cosmo_type = None,
                ) -> astro.astro_functions:
    """This function gets the old astro object aswell as a new set of
    parameters. It recomputes the cosmology if needed and then compares
    current and fiducial asto objects to find the updated one
    """

    current_astropars = current_astro.astroparams

    # Check if cosmology needs to be updated / was allready updated
    if updated_cosmo:
        if cosmopars == updated_cosmo.fullcosmoparams:
            cosmo = updated_cosmo
        else:
            raise ValueError("The passed cosmopars do not match the"
                             + "cosmopars in the updated comsology")
    else:
        cosmo = update_cosmo(current_cosmo, cosmopars)

    # handle trivial cases
    if (astropars == cfg.fiducialastroparams
            and cosmopars == cfg.fiducialfullcosmoparams):
        astro_object = cfg.fiducialastro
    elif (astropars == current_astro.astroparams
            and cosmopars == current_astro.cosmopars):
        astro_object = current_astro
    else:
        # The cosmology was allready updated in the first block
        astro_object = astro.astro_functions(cosmopars=cosmopars,
                                             astropars=astropars,
                                             cosmology=cosmo,
                                             )
    return astro_object