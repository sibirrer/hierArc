"""
this file contains routines to process the data format of the SLACS IFU data set to be binned in radial annuli
and for error estimates
"""
import numpy as np
import astropy.io.fits as fits


def radial_dispersion(dispersion_map, weight_map, flux_map, fibre_scale):
    """
    :param dispersion_map: velocity dispersion map
    :param flux_map: surface brightness map
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
            r = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) * fibre_scale
            disp = dispersion_map[i, j]
            if np.isfinite(disp):  # and disp < 800:
                r_list.append(r)
                disp_list.append(disp)
                weight_list.append(weight_map[i, j])
                flux_list.append(flux_map[i, j])
    r_list, disp_list, weight_list, flux_list = np.array(r_list), np.array(disp_list), np.array(weight_list), np.array(
        flux_list)
    return r_list, disp_list, weight_list, flux_list


def binning_dispersion(r_list, disp_list, weight_list, flux_list, r_bins):
    """
    """
    r_in = r_bins[0]
    disp_r = []
    weight_r = []

    # set nan to zeros
    for i, r_out in enumerate(r_bins[1:]):
        index = np.where((r_list >= r_in) & (r_list < r_out))
        disp = np.sum(disp_list[index] * weight_list[index] * flux_list[index]) / np.sum(
            weight_list[index] * flux_list[index])
        disp_r.append(disp)
        weight_r.append(np.sum(weight_list[index] * flux_list[index]) / np.sum(flux_list[index]))
        r_in = r_out
    return np.array(disp_r), np.array(weight_r)


# dispersion from individual velocity measurements
def binning_velocity(r_list, vel_list, weight_list, flux_list, r_bins):
    r_in = r_bins[0]
    disp_r = []
    weight_r = []

    # set nan to zeros
    for i, r_out in enumerate(r_bins[1:]):
        index = np.where((r_list >= r_in) & (r_list < r_out))
        disp2 = np.sum(vel_list[index] ** 2 * weight_list[index] * flux_list[index]) / np.sum(
            weight_list[index] * flux_list[index])
        disp_r.append(np.sqrt(disp2))
        weight_r.append(np.sum(weight_list[index] * flux_list[index]) / np.sum(flux_list[index]))
        r_in = r_out
    return np.array(disp_r), np.array(weight_r)


def binned_data(file_path, r_bins, fibre_scale):
    """
    returned binned velocity dispersion data
    """
    #file_path = os.path.join(path2data, name + data_file_ending)
    # print(file_path)
    hdul = fits.open(file_path)
    velocity = hdul[1].data
    dispersion = hdul[2].data
    surface_brightness = hdul[3].data
    sn_ration = hdul[4].data
    d_v_low = hdul[5].data
    d_v_high = hdul[6].data
    d_v_mean = (d_v_high - d_v_low) / 2
    d_sigma_low = hdul[7].data
    d_sigma_high = hdul[8].data
    d_sigma_mean = (d_sigma_high - d_sigma_low) / 2

    r_list, disp_list, weight_list, flux_list = radial_dispersion(dispersion, 1. / d_sigma_mean ** 2,
                                                                  surface_brightness, fibre_scale)
    disp_r, weight_r = binning_dispersion(r_list, disp_list, weight_list, flux_list, r_bins)
    r_list, v_list, weight_v_list, flux_list = radial_dispersion(velocity, 1. / d_v_mean ** 2, surface_brightness,
                                                                 fibre_scale)
    disp_v, weight_r_v = binning_velocity(r_list, v_list, weight_v_list, flux_list, r_bins)
    disp_tot = np.sqrt(
        disp_v ** 2 + disp_r ** 2)  # total measured dispersion integrated over velocity and local dispersion
    weight_tot = (weight_r * disp_r ** 2 + weight_r_v * disp_v ** 2) / disp_tot ** 2
    disp_error = 1 / np.sqrt(weight_tot)
    return disp_tot, disp_error

