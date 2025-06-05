import numpy as np

###
# The expressions presented here are generated in
#   generate_expressions/NoRadicals.nb
###


# 4 Halo terms
def T3111_kernel(k1, k2, b1k1, b1k2, b2k2, bG2k2, b3k2, bdG2k2, bG3k2, bDG2k2):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return (
        np.power(b1k1, 2)
        * b1k2
        * (
            -4
            * k1
            * k2
            * (
                b1k2
                * (
                    21 * np.power(k1, 6)
                    - 50 * np.power(k1, 4) * np.power(k2, 2)
                    + 79 * np.power(k1, 2) * np.power(k2, 4)
                    - 6 * np.power(k2, 6)
                )
                - 12
                * bDG2k2
                * (
                    3 * np.power(k1, 6)
                    - 11 * np.power(k1, 4) * np.power(k2, 2)
                    - 11 * np.power(k1, 2) * np.power(k2, 4)
                    + 3 * np.power(k2, 6)
                )
                + 12
                * np.power(k1, 2)
                * (
                    np.power(k2, 2)
                    * (
                        -68 * b2k2 * np.power(k1, 2)
                        - 21 * b3k2 * np.power(k1, 2)
                        + 56 * bdG2k2 * np.power(k1, 2)
                        + 98 * bG2k2 * np.power(k1, 2)
                        + 42 * bG2k2 * np.power(k2, 2)
                    )
                    + 28
                    * bG3k2
                    * (
                        2 * np.power(k1, 4)
                        - 3 * np.power(k1, 2) * np.power(k2, 2)
                        + np.power(k2, 4)
                    )
                )
            )
            + 6
            * np.power(np.power(k1, 2) - np.power(k2, 2), 2)
            * (
                b1k2
                * (
                    7 * np.power(k1, 4)
                    - 5 * np.power(k1, 2) * np.power(k2, 2)
                    - 2 * np.power(k2, 4)
                )
                - 12
                * (
                    -14 * bG2k2 * np.power(k1, 2) * np.power(k2, 2)
                    + bDG2k2 * np.power(np.power(k1, 2) - np.power(k2, 2), 2)
                )
            )
            * np.log(((k1 + k2) / np.abs(k1 - k2)))
        )
        / (6048.0 * np.power(k1, 5) * np.power(k2, 3))
    )


def T3111_squeezed(b1k1, b2k1, bG2k1, b3k1, bdG2k1, bG3k1, bDG2k1):
    return (
        -(
            np.power(b1k1, 3)
            * (
                11 * b1k1
                - 204 * b2k1
                - 63 * b3k1
                + 168 * bdG2k1
                + 48 * bDG2k1
                + 420 * bG2k1
            )
        )
        / 378
    )


def T2211_A_kernel(k1, k2, b1k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return (
        np.power(b1k1, 2)
        * (
            np.power(k1 + k2, an)
            * (
                np.power(b1k2, 2)
                * (
                    2400 * np.power(k1, 6)
                    - 2400 * an * np.power(k1, 5) * k2
                    + 360
                    * (-2 + an + 3 * np.power(an, 2))
                    * np.power(k1, 4)
                    * np.power(k2, 2)
                    - 40
                    * an
                    * (2 + 9 * an + 7 * np.power(an, 2))
                    * np.power(k1, 3)
                    * np.power(k2, 3)
                    + (
                        -1392
                        - 340 * an
                        + 478 * np.power(an, 2)
                        + 210 * np.power(an, 3)
                        + 49 * np.power(an, 4)
                    )
                    * np.power(k1, 2)
                    * np.power(k2, 4)
                    - 2
                    * an
                    * (-144 + 70 * an + 49 * np.power(an, 2))
                    * k1
                    * np.power(k2, 5)
                    + 2 * (-144 + 70 * an + 49 * np.power(an, 2)) * np.power(k2, 6)
                )
                - 14
                * b1k2
                * (k1 + k2)
                * (
                    (-12 + 4 * an + np.power(an, 2))
                    * b2k2
                    * np.power(k1, 2)
                    * (
                        20 * np.power(k1, 3)
                        - 20 * (1 + an) * np.power(k1, 2) * k2
                        + (8 + 15 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 2)
                        - (8 + 7 * an) * np.power(k2, 3)
                    )
                    + 4
                    * bG2k2
                    * (
                        120 * np.power(k1, 5)
                        - 120 * (1 + an) * np.power(k1, 4) * k2
                        + (42 + 89 * an + 47 * np.power(an, 2))
                        * np.power(k1, 3)
                        * np.power(k2, 2)
                        - (42 + 51 * an + 16 * np.power(an, 2) + 7 * np.power(an, 3))
                        * np.power(k1, 2)
                        * np.power(k2, 3)
                        + 3 * (2 + 9 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 4)
                        - 3 * (2 + 7 * an) * np.power(k2, 5)
                    )
                )
                + 49
                * np.power(k1 + k2, 2)
                * (
                    np.power(an, 4) * np.power(b2k2, 2) * np.power(k1, 4)
                    + 8
                    * np.power(an, 3)
                    * b2k2
                    * np.power(k1, 3)
                    * (b2k2 * k1 - bG2k2 * k2)
                    + 96
                    * bG2k2
                    * np.power(k1 - k2, 2)
                    * (
                        -(b2k2 * np.power(k1, 2))
                        + bG2k2 * (np.power(k1, 2) + np.power(k2, 2))
                    )
                    + 4
                    * np.power(an, 2)
                    * np.power(k1, 2)
                    * (
                        np.power(b2k2, 2) * np.power(k1, 2)
                        + 8 * np.power(bG2k2, 2) * np.power(k2, 2)
                        + 2
                        * b2k2
                        * bG2k2
                        * (np.power(k1, 2) - 6 * k1 * k2 + np.power(k2, 2))
                    )
                    - 16
                    * an
                    * k1
                    * (
                        3 * np.power(b2k2, 2) * np.power(k1, 3)
                        - 2
                        * b2k2
                        * bG2k2
                        * k1
                        * (np.power(k1, 2) + k1 * k2 + np.power(k2, 2))
                        + 2
                        * np.power(bG2k2, 2)
                        * k2
                        * (3 * np.power(k1, 2) - 4 * k1 * k2 + 3 * np.power(k2, 2))
                    )
                )
            )
            - (
                np.power(b1k2, 2)
                * (
                    2400 * np.power(k1, 6)
                    + 2400 * an * np.power(k1, 5) * k2
                    + 360
                    * (-2 + an + 3 * np.power(an, 2))
                    * np.power(k1, 4)
                    * np.power(k2, 2)
                    + 40
                    * an
                    * (2 + 9 * an + 7 * np.power(an, 2))
                    * np.power(k1, 3)
                    * np.power(k2, 3)
                    + (
                        -1392
                        - 340 * an
                        + 478 * np.power(an, 2)
                        + 210 * np.power(an, 3)
                        + 49 * np.power(an, 4)
                    )
                    * np.power(k1, 2)
                    * np.power(k2, 4)
                    + 2
                    * an
                    * (-144 + 70 * an + 49 * np.power(an, 2))
                    * k1
                    * np.power(k2, 5)
                    + 2 * (-144 + 70 * an + 49 * np.power(an, 2)) * np.power(k2, 6)
                )
                - 14
                * b1k2
                * (k1 - k2)
                * (
                    (-12 + 4 * an + np.power(an, 2))
                    * b2k2
                    * np.power(k1, 2)
                    * (
                        20 * np.power(k1, 3)
                        + 20 * (1 + an) * np.power(k1, 2) * k2
                        + (8 + 15 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 2)
                        + (8 + 7 * an) * np.power(k2, 3)
                    )
                    + 4
                    * bG2k2
                    * (
                        120 * np.power(k1, 5)
                        + 120 * (1 + an) * np.power(k1, 4) * k2
                        + (42 + 89 * an + 47 * np.power(an, 2))
                        * np.power(k1, 3)
                        * np.power(k2, 2)
                        + (42 + 51 * an + 16 * np.power(an, 2) + 7 * np.power(an, 3))
                        * np.power(k1, 2)
                        * np.power(k2, 3)
                        + 3 * (2 + 9 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 4)
                        + 3 * (2 + 7 * an) * np.power(k2, 5)
                    )
                )
                + 49
                * np.power(k1 - k2, 2)
                * (
                    np.power(an, 4) * np.power(b2k2, 2) * np.power(k1, 4)
                    + 8
                    * np.power(an, 3)
                    * b2k2
                    * np.power(k1, 3)
                    * (b2k2 * k1 + bG2k2 * k2)
                    + 96
                    * bG2k2
                    * np.power(k1 + k2, 2)
                    * (
                        -(b2k2 * np.power(k1, 2))
                        + bG2k2 * (np.power(k1, 2) + np.power(k2, 2))
                    )
                    + 4
                    * np.power(an, 2)
                    * np.power(k1, 2)
                    * (
                        np.power(b2k2, 2) * np.power(k1, 2)
                        + 8 * np.power(bG2k2, 2) * np.power(k2, 2)
                        + 2
                        * b2k2
                        * bG2k2
                        * (np.power(k1, 2) + 6 * k1 * k2 + np.power(k2, 2))
                    )
                    - 16
                    * an
                    * k1
                    * (
                        3 * np.power(b2k2, 2) * np.power(k1, 3)
                        - 2
                        * b2k2
                        * bG2k2
                        * k1
                        * (np.power(k1, 2) - k1 * k2 + np.power(k2, 2))
                        - 2
                        * np.power(bG2k2, 2)
                        * k2
                        * (3 * np.power(k1, 2) + 4 * k1 * k2 + 3 * np.power(k2, 2))
                    )
                )
            )
            * np.power(np.abs(k1 - k2), an)
        )
    ) / (392.0 * (-2 + an) * an * (2 + an) * (4 + an) * (6 + an) * np.power(k1, 5) * k2)


def T2211_A_squeezed(k1, b1k1, b2k1, bG2k1, an):
    k1 = np.asarray(k1, dtype=complex)

    return (
        np.power(2, -3 + an)
        * np.power(b1k1, 2)
        * (
            (1016 + 7 * an * (-10 + 7 * an)) * np.power(b1k1, 2)
            - 28 * b1k1 * ((6 + an) * (-12 + 7 * an) * b2k1 + 4 * (38 - 7 * an) * bG2k1)
            + 196
            * (
                (4 + an) * (6 + an) * np.power(b2k1, 2)
                - 8 * (6 + an) * b2k1 * bG2k1
                + 32 * np.power(bG2k1, 2)
            )
        )
        * np.power(k1, an)
    ) / (49.0 * (2 + an) * (4 + an) * (6 + an))


def T2211_X_kernel(k1, k2, b1k1, b2k1, bG2k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return (
        b1k1
        * b1k2
        * (
            np.power(k1 + k2, an)
            * (
                -(
                    b1k1
                    * b1k2
                    * (
                        60 * (2 + 7 * an) * np.power(k1, 6)
                        - 60 * an * (2 + 7 * an) * np.power(k1, 5) * k2
                        + (
                            -120
                            + 676 * an
                            + 250 * np.power(an, 2)
                            + 189 * np.power(an, 3)
                        )
                        * np.power(k1, 4)
                        * np.power(k2, 2)
                        - an
                        * (
                            -80
                            + 816 * an
                            + 210 * np.power(an, 2)
                            + 49 * np.power(an, 3)
                        )
                        * np.power(k1, 3)
                        * np.power(k2, 3)
                        + (
                            -120
                            + 676 * an
                            + 250 * np.power(an, 2)
                            + 189 * np.power(an, 3)
                        )
                        * np.power(k1, 2)
                        * np.power(k2, 4)
                        - 60 * an * (2 + 7 * an) * k1 * np.power(k2, 5)
                        + 60 * (2 + 7 * an) * np.power(k2, 6)
                    )
                )
                + 7
                * b1k1
                * (k1 + k2)
                * (
                    (-12 + 4 * an + np.power(an, 2))
                    * b2k2
                    * np.power(k1, 2)
                    * (
                        (8 + 7 * an) * np.power(k1, 3)
                        - (8 + 15 * an + 7 * np.power(an, 2)) * np.power(k1, 2) * k2
                        + 20 * (1 + an) * k1 * np.power(k2, 2)
                        - 20 * np.power(k2, 3)
                    )
                    + 4
                    * bG2k2
                    * (
                        3 * (2 + 7 * an) * np.power(k1, 5)
                        - 3 * (2 + 9 * an + 7 * np.power(an, 2)) * np.power(k1, 4) * k2
                        + (42 + 51 * an + 16 * np.power(an, 2) + 7 * np.power(an, 3))
                        * np.power(k1, 3)
                        * np.power(k2, 2)
                        - (42 + 89 * an + 47 * np.power(an, 2))
                        * np.power(k1, 2)
                        * np.power(k2, 3)
                        + 120 * (1 + an) * k1 * np.power(k2, 4)
                        - 120 * np.power(k2, 5)
                    )
                )
                - 7
                * (k1 + k2)
                * (
                    b1k2
                    * (
                        (-12 + 4 * an + np.power(an, 2))
                        * b2k1
                        * np.power(k2, 2)
                        * (
                            20 * np.power(k1, 3)
                            - 20 * (1 + an) * np.power(k1, 2) * k2
                            + (8 + 15 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 2)
                            - (8 + 7 * an) * np.power(k2, 3)
                        )
                        + 4
                        * bG2k1
                        * (
                            120 * np.power(k1, 5)
                            - 120 * (1 + an) * np.power(k1, 4) * k2
                            + (42 + 89 * an + 47 * np.power(an, 2))
                            * np.power(k1, 3)
                            * np.power(k2, 2)
                            - (
                                42
                                + 51 * an
                                + 16 * np.power(an, 2)
                                + 7 * np.power(an, 3)
                            )
                            * np.power(k1, 2)
                            * np.power(k2, 3)
                            + 3
                            * (2 + 9 * an + 7 * np.power(an, 2))
                            * k1
                            * np.power(k2, 4)
                            - 3 * (2 + 7 * an) * np.power(k2, 5)
                        )
                    )
                    - 7
                    * (k1 + k2)
                    * (
                        (-12 + 4 * an + np.power(an, 2))
                        * b2k2
                        * np.power(k1, 2)
                        * (
                            an * (4 + an) * b2k1 * np.power(k2, 2)
                            + 4
                            * bG2k1
                            * (np.power(k1, 2) - (2 + an) * k1 * k2 + np.power(k2, 2))
                        )
                        + 4
                        * bG2k2
                        * (
                            (-12 + 4 * an + np.power(an, 2))
                            * b2k1
                            * np.power(k2, 2)
                            * (np.power(k1, 2) - (2 + an) * k1 * k2 + np.power(k2, 2))
                            + 8
                            * bG2k1
                            * (
                                3 * np.power(k1, 4)
                                - 3 * (2 + an) * np.power(k1, 3) * k2
                                + (6 + 4 * an + np.power(an, 2))
                                * np.power(k1, 2)
                                * np.power(k2, 2)
                                - 3 * (2 + an) * k1 * np.power(k2, 3)
                                + 3 * np.power(k2, 4)
                            )
                        )
                    )
                )
            )
            + (
                b1k1
                * (
                    b1k2
                    * (
                        60 * (2 + 7 * an) * np.power(k1, 6)
                        + 60 * an * (2 + 7 * an) * np.power(k1, 5) * k2
                        + (
                            -120
                            + 676 * an
                            + 250 * np.power(an, 2)
                            + 189 * np.power(an, 3)
                        )
                        * np.power(k1, 4)
                        * np.power(k2, 2)
                        + an
                        * (
                            -80
                            + 816 * an
                            + 210 * np.power(an, 2)
                            + 49 * np.power(an, 3)
                        )
                        * np.power(k1, 3)
                        * np.power(k2, 3)
                        + (
                            -120
                            + 676 * an
                            + 250 * np.power(an, 2)
                            + 189 * np.power(an, 3)
                        )
                        * np.power(k1, 2)
                        * np.power(k2, 4)
                        + 60 * an * (2 + 7 * an) * k1 * np.power(k2, 5)
                        + 60 * (2 + 7 * an) * np.power(k2, 6)
                    )
                    - 7
                    * (k1 - k2)
                    * (
                        (-12 + 4 * an + np.power(an, 2))
                        * b2k2
                        * np.power(k1, 2)
                        * (
                            (8 + 7 * an) * np.power(k1, 3)
                            + (8 + 15 * an + 7 * np.power(an, 2)) * np.power(k1, 2) * k2
                            + 20 * (1 + an) * k1 * np.power(k2, 2)
                            + 20 * np.power(k2, 3)
                        )
                        + 4
                        * bG2k2
                        * (
                            3 * (2 + 7 * an) * np.power(k1, 5)
                            + 3
                            * (2 + 9 * an + 7 * np.power(an, 2))
                            * np.power(k1, 4)
                            * k2
                            + (
                                42
                                + 51 * an
                                + 16 * np.power(an, 2)
                                + 7 * np.power(an, 3)
                            )
                            * np.power(k1, 3)
                            * np.power(k2, 2)
                            + (42 + 89 * an + 47 * np.power(an, 2))
                            * np.power(k1, 2)
                            * np.power(k2, 3)
                            + 120 * (1 + an) * k1 * np.power(k2, 4)
                            + 120 * np.power(k2, 5)
                        )
                    )
                )
                + 7
                * (k1 - k2)
                * (
                    b1k2
                    * (
                        (-12 + 4 * an + np.power(an, 2))
                        * b2k1
                        * np.power(k2, 2)
                        * (
                            20 * np.power(k1, 3)
                            + 20 * (1 + an) * np.power(k1, 2) * k2
                            + (8 + 15 * an + 7 * np.power(an, 2)) * k1 * np.power(k2, 2)
                            + (8 + 7 * an) * np.power(k2, 3)
                        )
                        + 4
                        * bG2k1
                        * (
                            120 * np.power(k1, 5)
                            + 120 * (1 + an) * np.power(k1, 4) * k2
                            + (42 + 89 * an + 47 * np.power(an, 2))
                            * np.power(k1, 3)
                            * np.power(k2, 2)
                            + (
                                42
                                + 51 * an
                                + 16 * np.power(an, 2)
                                + 7 * np.power(an, 3)
                            )
                            * np.power(k1, 2)
                            * np.power(k2, 3)
                            + 3
                            * (2 + 9 * an + 7 * np.power(an, 2))
                            * k1
                            * np.power(k2, 4)
                            + 3 * (2 + 7 * an) * np.power(k2, 5)
                        )
                    )
                    - 7
                    * (k1 - k2)
                    * (
                        (-12 + 4 * an + np.power(an, 2))
                        * b2k2
                        * np.power(k1, 2)
                        * (
                            an * (4 + an) * b2k1 * np.power(k2, 2)
                            + 4
                            * bG2k1
                            * (np.power(k1, 2) + (2 + an) * k1 * k2 + np.power(k2, 2))
                        )
                        + 4
                        * bG2k2
                        * (
                            (-12 + 4 * an + np.power(an, 2))
                            * b2k1
                            * np.power(k2, 2)
                            * (np.power(k1, 2) + (2 + an) * k1 * k2 + np.power(k2, 2))
                            + 8
                            * bG2k1
                            * (
                                3 * np.power(k1, 4)
                                + 3 * (2 + an) * np.power(k1, 3) * k2
                                + (6 + 4 * an + np.power(an, 2))
                                * np.power(k1, 2)
                                * np.power(k2, 2)
                                + 3 * (2 + an) * k1 * np.power(k2, 3)
                                + 3 * np.power(k2, 4)
                            )
                        )
                    )
                )
            )
            * np.power(np.abs(k1 - k2), an)
        )
    ) / (
        392.0
        * (-2 + an)
        * an
        * (2 + an)
        * (4 + an)
        * (6 + an)
        * np.power(k1, 3)
        * np.power(k2, 3)
    )


def T2211_X_squeezed(k1, b1k1, b2k1, bG2k1, an):
    k1 = np.asarray(k1, dtype=complex)

    return (
        np.power(2, -3 + an)
        * np.power(b1k1, 2)
        * (
            (1016 + 7 * an * (-10 + 7 * an)) * np.power(b1k1, 2)
            - 28 * b1k1 * ((6 + an) * (-12 + 7 * an) * b2k1 + 4 * (38 - 7 * an) * bG2k1)
            + 196
            * (
                (4 + an) * (6 + an) * np.power(b2k1, 2)
                - 8 * (6 + an) * b2k1 * bG2k1
                + 32 * np.power(bG2k1, 2)
            )
        )
        * np.power(k1, an)
    ) / (49.0 * (2 + an) * (4 + an) * (6 + an))


# 3 Halo terms


def T211_A_kernel(b1k1, b1k2, L2b1, L2b2, L2bG2):
    return ((34 * L2b1 + 21 * L2b2 - 28 * L2bG2) * b1k1 * b1k2) / 21.0


def T211_X_kernel(k1, k2, Lb1, Lb2, LbG2, L2b1, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return -(
        np.power((k1 + k2) / (np.power(k1, 3) * k2 + k1 * np.power(k2, 3)), an)
        * L2b1
        * Lb1
        * (
            np.power(k1 * k2 * (np.power(k1, 2) + np.power(k2, 2)), an)
            * (
                np.power(k1, 2)
                * np.power(k2, 2)
                * (
                    (-12 - 5 * an + 7 * np.power(an, 2)) * Lb1
                    - 7 * (np.power(an, 2) * Lb2 + 4 * an * (Lb2 - 2 * LbG2) - 8 * LbG2)
                )
                - 2
                * an
                * np.power(k1, 3)
                * k2
                * (10 * Lb1 + 7 * (4 + an) * Lb2 - 14 * LbG2)
                + an * k1 * np.power(k2, 3) * ((8 + 7 * an) * Lb1 + 28 * LbG2)
                - np.power(k2, 4) * ((8 + 7 * an) * Lb1 + 28 * LbG2)
                + np.power(k1, 4)
                * (20 * Lb1 - 7 * (4 * an * Lb2 + np.power(an, 2) * Lb2 + 4 * LbG2))
            )
            - (k1 - k2)
            * np.power((k1 * k2 * (np.power(k1, 2) + np.power(k2, 2))) / (k1 + k2), an)
            * (
                (1 + an) * k1 * np.power(k2, 2) * ((8 + 7 * an) * Lb1 + 28 * LbG2)
                + np.power(k2, 3) * ((8 + 7 * an) * Lb1 + 28 * LbG2)
                + np.power(k1, 3)
                * (20 * Lb1 - 7 * (4 * an * Lb2 + np.power(an, 2) * Lb2 + 4 * LbG2))
                + np.power(k1, 2)
                * k2
                * (
                    20 * (1 + an) * Lb1
                    + 7
                    * (4 * an * Lb2 + np.power(an, 2) * Lb2 - 4 * LbG2 - 4 * an * LbG2)
                )
            )
            * np.power(np.abs(k1 - k2), an)
        )
    ) / (14 * an * (2 + an) * (4 + an) * np.power(k1, 3) * k2)


def T211_X_squeezed(k1, Lb1, Lb2, LbG2, L2b1, an):
    k1 = np.asarray(k1, dtype=complex)

    return -(
        np.power(2, an)
        * np.power(k1, an)
        * L2b1
        * Lb1
        * ((-12 + 7 * an) * Lb1 - 14 * (4 + an) * Lb2 + 56 * LbG2)
    ) / (7 * (2 + an) * (4 + an))


# 2 Halo terms


def T22_kernel(k1, k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return (
        (np.power(k1 + k2, an + 2) - np.power(np.abs(k1 - k2), an + 2))
        / (2 * k1 * k2 * (an + 2))
    )
