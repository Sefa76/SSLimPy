""" The functions here will update the different cosmology and survey classes using a ladder system.
All constructors of the classes get as an argument an Object of the previous classes as well as new parameters.
Like this the update is done heirachially from bottom to top.
"""

from copy import copy

from SSLimPy.interface.config import Configuration
from SSLimPy.cosmology import cosmology
from SSLimPy.cosmology import halo_model
from SSLimPy.cosmology import astro
from SSLimPy.interface import survey_specs


def update_surveySpecs(
    obspars: dict,
    cfg: Configuration,
) -> survey_specs.SurveySpecifications:
    """This class only depends on the fiducial cosmology and
    needs only one dict
    """
    return survey_specs.SurveySpecifications(obspars, cfg.fiducialcosmo)


def update_cosmo(
    current_cosmo: cosmology.CosmoFunctions,
    cosmopars: dict,
    cfg: Configuration,
) -> cosmology.CosmoFunctions:
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
            cosmo = cosmology.CosmoFunctions(cfg, cosmology=current_cosmo, nuiscance_like=nuiscance_like)
    elif cosmopars == cfg.fiducialcosmoparams:
        cosmo = cfg.fiducialcosmo
        if nuiscance_like != cfg.fiducialnuiscancelikeparams:
            cosmo = cosmology.CosmoFunctions(cfg, cosmology=cfg.fiducialcosmo, nuiscance_like=nuiscance_like)
    else:
        cosmo = cosmology.CosmoFunctions(cfg, cosmopars, nuiscance_like)
    return cosmo


def update_halomodel(
    current_halomodel: halo_model.HaloModel,
    cosmopars: dict,
    halopars: dict,
    cfg: Configuration,
) -> halo_model.HaloModel:
    """Will update the cosmology and halomodel with passed parameters."""
    current_cosmo = current_halomodel.cosmology
    current_haloparams = current_halomodel.haloparams

    updated_cosmo = update_cosmo(current_cosmo, cosmopars, cfg)

    if updated_cosmo == current_cosmo:
        if halopars == current_haloparams:
            halo = current_halomodel
        else:
            halo = halo_model.HaloModel(current_cosmo, halopars)
    elif updated_cosmo == cfg.fiducialcosmo:
        if halopars == cfg.fiducialhaloparams:
            halo = cfg.fiducialhalomodel
        else:
            halo = halo_model.HaloModel(cfg.fiducialcosmo, halopars)
    else:
        halo = halo_model.HaloModel(updated_cosmo, halopars)
    return halo


def update_astro(
    current_astro: astro.AstroFunctions,
    cosmopars: dict,
    halopars: dict,
    astropars: dict,
    obspars: dict,
    cfg: Configuration,
) -> astro.AstroFunctions:
    """Will update the cosmology, halomodel, survey specifications, and the astro model with passed parameters."""

    current_halomodel = current_astro.halomodel
    current_specs = current_astro.survey_specs
    current_astoparams = current_astro.astroparams

    updated_halomodel = update_halomodel(current_halomodel, cosmopars, halopars, cfg)

    if updated_halomodel == current_halomodel:
        if astropars == current_astoparams:
            if obspars == current_specs.obsparams:
                return current_astro
            else:
                survey_specs = update_surveySpecs(obspars, cfg)
                return astro.AstroFunctions(current_halomodel, survey_specs, current_astoparams)
        else:
            survey_specs = update_surveySpecs(obspars, cfg)
            return astro.AstroFunctions(current_halomodel, survey_specs, astropars)
    elif updated_halomodel == cfg.fiducialhalomodel:
        if astropars == cfg.fiducialastroparams:
            if obspars == cfg.fiducialspecparams:
                return cfg.fiducialastro
            else:
                survey_specs = update_surveySpecs(obspars, cfg)
                return astro.AstroFunctions(cfg.fiducialhalomodel, survey_specs, cfg.fiducialastroparams)
        else:
            survey_specs = update_surveySpecs(obspars, cfg)
            return astro.AstroFunctions(cfg.fiducialhalomodel, survey_specs, astropars)
    else:
        survey_specs = update_surveySpecs(obspars, cfg)
        return astro.AstroFunctions(updated_halomodel, survey_specs, astropars)
