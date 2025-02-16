""" The functions here will update the different cosmology and survey classes using a ladder system.
All constructors of the classes get as an argument an Object of the previous classes as well as new parameters.
Like this the update is done heirachially from bottom to top.
"""

from copy import copy

import SSLimPy.interface.config as cfg
from SSLimPy.cosmology import cosmology, halomodel, astro
from SSLimPy.interface import surveySpecs


def update_surveySpecs(
    obspars: dict,
) -> surveySpecs.survey_specifications:
    """This class only depends on the fiducial cosmology and
    needs only one dict
    """
    return surveySpecs.survey_specifications(obspars, cfg.fiducialcosmo)


def update_cosmo(
    current_cosmo: cosmology.cosmo_functions,
    cosmopars: dict,
) -> cosmology.cosmo_functions:
    """This function gets the old cosmology aswell as a new set of
    cosmological parameters. It then seperates EBS parameters from
    nuiscance parameters. Finally it compares the inputs to previously
    computed cosmology results to avoid recomputation
    """
    current_cosmopars = current_cosmo.cosmopars
    current_nuiscance_like = current_cosmo.nuiscance_like

    # seperate nuiscance-like cosmo pars
    cosmopars = copy(cosmopars)
    nuiscance_like = dict()
    for key in cosmopars:
        if key in cfg.nuiscance_like_params_names:
            nuiscance_like[key] = cosmopars.pop(key)

    if cosmopars == current_cosmopars:
        cosmo = current_cosmo
        if nuiscance_like != current_nuiscance_like:
            cosmo = cosmology.cosmo_functions(
                cosmology=current_cosmo, nuiscance_like=nuiscance_like
            )
    elif cosmopars == cfg.fiducialcosmoparams:
        cosmo = cfg.fiducialcosmo
        if nuiscance_like != cfg.fiducialnuiscancelikeparams:
            cosmo = cosmology.cosmo_functions(
                cosmology=cfg.fiducialcosmo, nuiscance_like=nuiscance_like
            )
    else:
        cosmo = cosmology.cosmo_functions(cosmopars, nuiscance_like)
    return cosmo


def update_halomodel(
    current_halomodel: halomodel.halomodel,
    cosmopars: dict,
    halopars: dict,
) -> halomodel.halomodel:
    """Will update the cosmology and halomodel with passed parameters."""
    current_cosmo = current_halomodel.cosmology
    current_haloparams = current_halomodel.haloparams

    updated_cosmo = update_cosmo(current_cosmo, cosmopars)

    if updated_cosmo == current_cosmo:
        if halopars == current_haloparams:
            halo = current_halomodel
        else:
            halo = halomodel.halomodel(current_cosmo, halopars)
    elif updated_cosmo == cfg.fiducialcosmo:
        if halopars == cfg.fiducialhaloparams:
            halo = cfg.fiducialhalomodel
        else:
            halo = halomodel.halomodel(cfg.fiducialcosmo, halopars)
    else:
        halo = halomodel.halomodel(updated_cosmo, halopars)
    return halo


def update_astro(
    current_astro: astro.astro_functions,
    cosmopars: dict,
    halopars: dict,
    astropars: dict,
    obspars: dict,
) -> astro.astro_functions:
    """Will update the cosmology, halomodel, survey specifications, and the astro model with passed parameters."""

    current_halomodel = current_astro.halomodel
    current_specs = current_astro.survey_specs
    current_astoparams = current_astro.astroparams

    updated_halomodel = update_halomodel(current_halomodel, cosmopars, halopars)

    if updated_halomodel == current_halomodel:
        if astropars == current_astoparams:
            if obspars == current_specs.obsparams:
                return current_astro
            else:
                survey_specs = update_surveySpecs(obspars)
                return astro.astro_functions(current_halomodel, survey_specs, current_astoparams)
        else:
            survey_specs = update_surveySpecs(obspars)
            return astro.astro_functions(current_halomodel, survey_specs, astropars)
    elif updated_halomodel == cfg.fiducialhalomodel:
        if astropars == cfg.fiducialastroparams:
            if obspars == cfg.fiducialspecparams:
                return cfg.fiducialastro
            else:
                survey_specs = update_surveySpecs(obspars)
                return astro.astro_functions(cfg.fiducialhalomodel, survey_specs, cfg.fiducialastroparams)
        else:
            survey_specs = update_surveySpecs(obspars)
            return astro.astro_functions(cfg.fiducialhalomodel, survey_specs, astropars)
    else:
        survey_specs = update_surveySpecs(obspars)
        return astro.astro_functions(updated_halomodel, survey_specs, astropars)
