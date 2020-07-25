from numpy.testing import assert_array_almost_equal


def assert_array_almost_equal_up_to_sign(actual, desired, decimal=6,
                    err_msg='', verbose=True):
    """Like numpy.test.assert_array_almost_equal, but modified to allow objects to have different signs
    """
    msg = ''

    try:
        assert_array_almost_equal(actual, desired, decimal, err_msg, verbose)
    except Exception as e_pos:
        error_positive = True
        msg += str(e_pos)
    else:
        error_positive = False

    try:
        assert_array_almost_equal(actual, -desired, decimal, err_msg, verbose)
    except Exception as e_neg:
        error_negative = True
        msg += '\t' + str(e_neg)
    else:
        error_negative = False

    if error_positive and error_negative:
        raise AssertionError(msg)