import numpy as np
import pytest
from hierarc.JAM.aperture import Aperture


class TestAperture:
    def setup_method(self):
        self.x_coords = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.y_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.bins = np.array([[0, 1], [2, -1]])
        self.aperture_kwargs = {
            "general": {"x_cords": np.array([0.0, 1.0, 2.0]), "y_cords": np.array([0.0, 0.5, 1.0]), "bin_ids": np.array([0, 1, 1]), "delta_pix": 0.2},
            "slit": {"length": 0.3, "width": 0.1, "center_ra": 0.0, "center_dec": 0.0, "angle": 0.0, "delta_pix": 0.1},
            "shell": {"r_in": 0.5, "r_out": 1.0, "center_ra": 0.0, "center_dec": 0.0, "delta_pix": 0.1},
            "frame": {"width_outer": 0.6, "width_inner": 0.2, "center_ra": 0.0, "center_dec": 0.0, "angle": 0.0, "delta_pix": 0.1},
            "IFU_grid": {"x_grid": self.x_coords, "y_grid": self.y_coords, "supersampling_factor": 1, "padding_arcsec": 0.1},
            "IFU_shells": {"r_bins": np.array([0.0, 0.5, 1.0]), "center_ra": 0.0, "center_dec": 0.0, "ifu_grid_kwargs": None, "delta_pix": 0.1},
            "IFU_binned": {"x_grid": self.x_coords, "y_grid": self.y_coords, "bins": self.bins},
        }

    def test_supported_types(self):
        """Construct Aperture for each supported aperture_type string and check basic forwarding."""
        supported = [
            "general",
            "slit",
            "shell",
            "frame",
            "IFU_grid",
            "IFU_shells",
            "IFU_binned",
        ]
        for a_type in supported:
            kw = self.aperture_kwargs[a_type]
            ap = Aperture(a_type, **kw)
            assert ap.aperture_type == a_type
            assert isinstance(int(ap.num_segments), int)
            sample = ap.aperture_sample()
            assert isinstance(sample, tuple) and len(sample) == 2
            hr_map = np.random.random(size=sample[0].shape)
            out = ap.aperture_downsample(hr_map)
            assert np.size(out) == ap.num_segments
            assert np.isfinite(ap.delta_pix)


class TestRaise:

    def test_invalid_aperture_type_raises(self):
        with pytest.raises(ValueError):
            Aperture("not_a_valid_type")

    def test_invalid_aperture_kwargs_raises(self):
        with pytest.raises(TypeError):
            Aperture("slit", invalid_arg=np.array([0.0, 1.0]))


if __name__ == "__main__":
    pytest.main()
