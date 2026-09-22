import numpy as np
import pytest

from afterburn_watch import indices, masking
from afterburn_watch.sensors import LANDSAT_C2_L2, SENTINEL2_L2A, get_sensor, scale_offset


def test_nbr_uses_correct_bands_per_sensor():
    # Keith's email: Landsat 8 NBR = (B5 - B7)/(B5 + B7); Sentinel-2 uses B08/B12
    assert SENTINEL2_L2A.bands["nir"].band_id == "B08"
    assert SENTINEL2_L2A.bands["swir2"].band_id == "B12"
    assert LANDSAT_C2_L2.bands["nir"].band_id == "B5"
    assert LANDSAT_C2_L2.bands["swir2"].band_id == "B7"
    assert LANDSAT_C2_L2.asset_for("nir") == "nir08"
    assert SENTINEL2_L2A.asset_for("swir2") == "swir22"


def test_unknown_sensor_and_role_raise():
    with pytest.raises(KeyError):
        get_sensor("modis")
    with pytest.raises(KeyError):
        SENTINEL2_L2A.asset_for("thermal")


def test_scale_offset_prefers_item_metadata():
    assets = {"swir22": {"raster:bands": [{"scale": 0.0001, "offset": -0.1}]}}
    assert scale_offset(assets, SENTINEL2_L2A, "swir2") == (0.0001, -0.1)
    assert scale_offset({}, LANDSAT_C2_L2, "nir") == (0.0000275, -0.2)


def test_indices_known_values():
    nir, swir2, red = np.array([0.5]), np.array([0.1]), np.array([0.1])
    np.testing.assert_allclose(indices.nbr(nir, swir2), [0.4 / 0.6], rtol=1e-6)
    np.testing.assert_allclose(indices.ndvi(nir, red), [0.4 / 0.6], rtol=1e-6)
    np.testing.assert_allclose(indices.dnbr(np.array([0.6]), np.array([0.1])), [500.0])
    assert indices.dndvi(np.array([0.7]), np.array([0.2]))[0] == pytest.approx(0.5)
    assert np.isnan(indices.nbr(np.array([0.0]), np.array([0.0]))[0])


def test_rdnbr_and_reflectance():
    out = indices.rdnbr(np.array([400.0, 400.0]), np.array([0.25, 0.0]))
    assert out[0] == pytest.approx(800.0)
    assert np.isnan(out[1])  # pre-fire NBR ~0 -> undefined
    np.testing.assert_allclose(indices.reflectance(np.array([10000]), 0.0001, -0.1), [0.9], rtol=1e-6)


def test_scl_mask_keeps_burn_scars_in_dark_class():
    scl = np.array([0, 2, 3, 4, 5, 8, 9, 10, 11])
    valid = masking.scl_valid_mask(scl)
    assert valid.tolist() == [False, True, False, True, True, False, False, False, True]


def test_qa_pixel_bits():
    clear = 1 << 6
    cloud = 1 << 3
    shadow = 1 << 4
    fill = 1 << 0
    qa = np.array([clear, cloud, shadow, fill, clear | (1 << 7)], dtype="uint16")
    assert masking.qa_pixel_valid_mask(qa).tolist() == [True, False, False, False, True]


def test_aerosol_masks():
    qa_aer = np.array([0 << 6, 1 << 6, 2 << 6, 3 << 6], dtype="uint8")
    assert masking.qa_aerosol_valid_mask(qa_aer).tolist() == [True, True, True, False]
    aot = np.array([100, 900])  # 0.1 and 0.9
    assert masking.aot_valid_mask(aot, max_aot=0.5).tolist() == [True, False]


def test_masked_median_composite_uses_only_clear_views():
    stack = np.array([[[1.0, 5.0]], [[3.0, 100.0]], [[2.0, 7.0]]])  # (time, 1, 2)
    valid = np.array([[[True, False]], [[True, False]], [[True, False]]])
    comp, n = masking.masked_median_composite(stack, valid)
    assert comp[0, 0] == pytest.approx(2.0)
    assert np.isnan(comp[0, 1]) and n[0, 1] == 0
    valid[1, 0, 1] = True
    comp, _ = masking.masked_median_composite(stack, valid)
    assert comp[0, 1] == pytest.approx(100.0)
