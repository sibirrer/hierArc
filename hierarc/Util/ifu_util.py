"""
this file contains routines to process the data format of the SLACS IFU data set to be binned in radial annuli
and for error estimates
"""
import numpy as np
import astropy.io.fits as fits


def binned_dispersion(dispersion_map, weight_map, flux_map, fiber_scale, r_bins):
    """

    :param dispersion_map: 2d array of measured velocity dispersion for each fiber
    :param weight_map: uncertainty weight of the measurement (i.e. 1/sigma**2) for each fiber
    :param flux_map: array of flux for each fiber
    :param fiber_scale: separation of the fibers (map pixels)
    :param r_bins: array, bins in radial directions
    :return: averaged velocity dispersion measurement for each radial bin and estimated uncertainty thereof
    """
    radial_pos, dispersion, error_weight, flux = _2d_t0_1d(dispersion_map, weight_map, flux_map, fiber_scale)
    r_in = r_bins[0]
    disp_r = []
    weight_r = []

    # set nan to zeros
    for i, r_out in enumerate(r_bins[1:]):
        index = np.where((radial_pos >= r_in) & (radial_pos < r_out))
        disp = np.sum(dispersion[index] * error_weight[index] * flux[index]) / np.sum(
            error_weight[index] * flux[index])
        disp_r.append(disp)
        weight_r.append(np.sum(error_weight[index] * flux[index]) / np.sum(flux[index]))
        r_in = r_out
    return np.array(disp_r), np.array(weight_r)


def binned_velocity(velocity_map, weight_map, flux_map, fiber_scale, r_bins):
    """

    :param velocity_map: 2d array of measured velocity offset for each fiber
    :param weight_map: uncertainty weight of the velocity measurement (i.e. 1/sigma**2) for each fiber
    :param flux_map: array of flux for each fiber
    :param fiber_scale: separation of the fibers (map pixels)
    :param r_bins: array, bins in radial directions
    :return: average velocity components with luminosity weights, weight on v^2 error
    """
    radial_pos, velocity, error_weight, flux = _2d_t0_1d(velocity_map, weight_map, flux_map, fiber_scale)
    r_in = r_bins[0]
    v2_r = []
    error_weight_v2 = error_weight / (2 * np.abs(velocity))  # error on v^2 based on error on v, 1/sigma^2(v^2) = 1/sigma^2(v) / (2 * v)
    weight_r = []
    for i, r_out in enumerate(r_bins[1:]):
        index = np.where((radial_pos >= r_in) & (radial_pos < r_out))
        v2 = np.sum(velocity[index] ** 2 * error_weight_v2[index] * flux[index]) / np.sum(
            error_weight_v2[index] * flux[index])
        v2_r.append(v2)
        weight_r.append(np.sum(error_weight_v2[index] * flux[index]) / np.sum(flux[index]))
        r_in = r_out
    v_r = np.sqrt(np.array(v2_r))
    weight_v = np.array(weight_r) * (2 * v_r)
    return v_r, weight_v


def binned_total(dispersion_map, weight_map_disp, velocity_map, weight_map_v, flux_map, fiber_scale, r_bins):
    """

    :param dispersion_map: 2d array of measured velocity dispersion for each fiber
    :param weight_map_disp: uncertainty weight of the measurement in velocity dispersion (i.e. 1/sigma**2) for each fiber
    :param velocity_map: 2d array of measured velocity offset for each fiber
    :param weight_map_v: uncertainty weight of the measurement in systemic velocity (i.e. 1/sigma**2) for each fiber
    :param flux_map: array of flux for each fiber
    :param fiber_scale: separation of the fibers (map pixels)
    :param r_bins: array, bins in radial directions
    :return: sqrt(v^2 + sigma^2) averaged integrated line dispersion when averaged over azimuthal bins
    """
    v_r, weight_v_r = binned_velocity(velocity_map, weight_map_v, flux_map, fiber_scale, r_bins)
    disp_r, weight_disp_r = binned_dispersion(dispersion_map, weight_map_disp, flux_map, fiber_scale, r_bins)

    disp_tot = np.sqrt(v_r ** 2 + disp_r ** 2)  # total measured dispersion integrated over velocity and local dispersion
    weight_tot = (weight_disp_r * disp_r ** 2 + weight_v_r * v_r ** 2) / disp_tot ** 2
    disp_error = 1 / np.sqrt(weight_tot)
    return disp_tot, disp_error


def _2d_t0_1d(value_map, weight_map, flux_map, fiber_scale):
    """
    transforms map in 1-d array removing nans and artifacts


    :param value_map: quantity (e.g. velocity dispersion or velocity) on the map level
    :param weight_map:
    :param flux_map: surface brightness map
    :param fiber_scale: separation of the fibers (map pixels)
    """
    # find center
    x_index, y_index = np.where(flux_map == np.max(flux_map))
    center_x, center_y = x_index[0], y_index[0]
    nx, ny = np.shape(flux_map)
    r_list = []
    disp_list = []
    weight_list = []
    flux_list = []
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) * fiber_scale
            disp = value_map[i, j]
            if np.isfinite(disp) and np.isfinite(weight_map[i, j]):  # and disp < 800:
                r_list.append(r)
                disp_list.append(disp)
                weight_list.append(weight_map[i, j])
                flux_list.append(flux_map[i, j])
    r_list, disp_list, weight_list, flux_list = np.array(r_list), np.array(disp_list), np.array(weight_list), np.array(
        flux_list)
    return r_list, disp_list, weight_list, flux_list
