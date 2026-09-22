import numpy as np

from afterburn_watch.severity import classify_baer, compute_dnbr, compute_nbr


def test_compute_nbr_basic():
    nir = np.array([0.5, 0.8])
    swir2 = np.array([0.1, 0.2])
    nbr = compute_nbr(nir, swir2)
    expected = np.array([(0.5 - 0.1) / (0.5 + 0.1), (0.8 - 0.2) / (0.8 + 0.2)])
    np.testing.assert_allclose(nbr, expected, rtol=1e-6)


def test_compute_nbr_handles_zero_denominator():
    nir = np.array([0.0])
    swir2 = np.array([0.0])
    nbr = compute_nbr(nir, swir2)
    assert np.isnan(nbr[0])


def test_compute_dnbr():
    nbr_pre = np.array([0.6])
    nbr_post = np.array([0.1])
    dnbr = compute_dnbr(nbr_pre, nbr_post)
    np.testing.assert_allclose(dnbr, [500.0])


def test_classify_baer_breaks():
    dnbr = np.array([50, 150, 400, 700, np.nan])
    classes = classify_baer(dnbr)
    np.testing.assert_array_equal(classes, [0, 1, 2, 3, -1])
