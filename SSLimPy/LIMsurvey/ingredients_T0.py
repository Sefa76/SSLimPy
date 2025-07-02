import numpy as np
Power = np.power
Abs = np.abs
Log = np.log

###
# The expressions presented here are generated in
#   generate_expressions/NoRadicals.nb
###


# 4 Halo terms
def T3111_kernel(k1, k2, b1k1, b1k2, b2k2, bG2k2, b3k2, bdG2k2, bG3k2, bDG2k2):
    return (Power(b1k1,2)*b1k2*(3264*b2k2*Power(k1,5)*Power(k2,3) +
       1008*b3k2*Power(k1,5)*Power(k2,3) - 2688*bdG2k2*Power(k1,5)*Power(k2,3) -
       1344*bG3k2*Power(k1,3)*k2*(2*Power(k1,4) - 3*Power(k1,2)*Power(k2,2) +
          Power(k2,4)) + 168*bG2k2*Power(k1,2)*Power(k2,2)*
        (-4*(7*Power(k1,3)*k2 + 3*k1*Power(k2,3)) -
          3*Power(Power(k1,2) - Power(k2,2),2)*(Log(Power(k1 - k2,2)) - 2*Log(k1 + k2)))\
        + bDG2k2*(48*k1*k2*(Power(k1,2) + Power(k2,2))*
           (3*Power(k1,4) - 14*Power(k1,2)*Power(k2,2) + 3*Power(k2,4)) +
          36*Power(Power(k1,2) - Power(k2,2),4)*(Log(Power(k1 - k2,2)) - 2*Log(k1 + k2)))
         + b1k2*(4*k1*k2*(-21*Power(k1,6) + 50*Power(k1,4)*Power(k2,2) -
             79*Power(k1,2)*Power(k2,4) + 6*Power(k2,6)) -
          3*Power(Power(k1,2) - Power(k2,2),3)*(7*Power(k1,2) + 2*Power(k2,2))*
           (Log(Power(k1 - k2,2)) - 2*Log(k1 + k2)))))/(6048.*Power(k1,5)*Power(k2,3))

def T3111_s_k2ok1(k1, k2, b1k1, b1k2, b2k2, bG2k2, b3k2, bdG2k2, bG3k2, bDG2k2):
    eps = k2/k1

    return (Power(b1k1,2)*b1k2*(1020*b2k2 + 315*b3k2 - 840*bdG2k2 - 61*b1k2*Power(eps,2) -
       384*bDG2k2*Power(eps,2) - 840*bG2k2*(1 + 2*Power(eps,2)) -
       (420*bG3k2*(2 - 3*Power(eps,2) + Power(eps,4)))/Power(eps,2)))/1890.

def T3111_s_k1ok2(k1, k2, b1k1, b1k2, b2k2, bG2k2, b3k2, bdG2k2, bG3k2, bDG2k2):
    eps = k1/k2

    return (Power(b1k1,2)*b1k2*(7140*b2k2 + 2205*b3k2 - 5880*bdG2k2 +
       b1k2*(812 - 735/Power(eps,2) - 564*Power(eps,2)) + 1176*bG2k2*(-15 + 2*Power(eps,2)) +
       384*bDG2k2*(-7 + 3*Power(eps,2)) -
       (2940*bG3k2*(1 - 3*Power(eps,2) + 2*Power(eps,4)))/Power(eps,2)))/13230.

def T3111_squeezed(k1, b1, b2, bG2, b3, bdG2, bG3, bDG2):
    return Power(b1,3)*((-11*b1)/378. + (34*b2)/63. + b3/6. - (4*bdG2)/9. - (8*bDG2)/63. - 
     (10*bG2)/9.)

def T2211_A_kernel(k1, k2, b1k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return (np.power(b1k1,2)*(-14*(-2 + an)*(6 + an)*b1k2*b2k2*np.power(k1,2)*
        (np.power(k1 + k2,1 + an)*(20*np.power(k1,3) - 20*(1 + an)*np.power(k1,2)*k2 +
             (1 + an)*(8 + 7*an)*k1*np.power(k2,2) - (8 + 7*an)*np.power(k2,3)) -
          (k1 - k2)*(20*np.power(k1,3) + 20*(1 + an)*np.power(k1,2)*k2 +
             (1 + an)*(8 + 7*an)*k1*np.power(k2,2) + (8 + 7*an)*np.power(k2,3))*
           np.power(np.abs(k1 - k2),an)) +
       b1k2*bG2k2*(-56*np.power(k1 + k2,1 + an)*
           (120*np.power(k1,5) - 120*(1 + an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + 47*an)*np.power(k1,3)*np.power(k2,2) -
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,2)*np.power(k2,3) +
             3*(1 + an)*(2 + 7*an)*k1*np.power(k2,4) - 3*(2 + 7*an)*np.power(k2,5)) +
          56*(k1 - k2)*(120*np.power(k1,5) + 120*(1 + an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + 47*an)*np.power(k1,3)*np.power(k2,2) +
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,2)*np.power(k2,3) +
             3*(1 + an)*(2 + 7*an)*k1*np.power(k2,4) + 3*(2 + 7*an)*np.power(k2,5))*
           np.power(np.abs(k1 - k2),an)) +
       np.power(b1k2,2)*(np.power(k1 + k2,an)*
           (2400*np.power(k1,6) - 2400*an*np.power(k1,5)*k2 +
             360*(-2 + an + 3*np.power(an,2))*np.power(k1,4)*np.power(k2,2) -
             40*an*(1 + an)*(2 + 7*an)*np.power(k1,3)*np.power(k2,3) +
             (-1392 + an*(-340 + an*(478 + 7*an*(30 + 7*an))))*np.power(k1,2)*np.power(k2,4) -
             2*an*(-8 + 7*an)*(18 + 7*an)*k1*np.power(k2,5) +
             2*(-8 + 7*an)*(18 + 7*an)*np.power(k2,6)) -
          (2400*np.power(k1,6) + 2400*an*np.power(k1,5)*k2 +
             360*(-2 + an + 3*np.power(an,2))*np.power(k1,4)*np.power(k2,2) +
             40*an*(1 + an)*(2 + 7*an)*np.power(k1,3)*np.power(k2,3) +
             (-1392 + an*(-340 + an*(478 + 7*an*(30 + 7*an))))*np.power(k1,2)*np.power(k2,4) +
             2*an*(-8 + 7*an)*(18 + 7*an)*k1*np.power(k2,5) +
             2*(-8 + 7*an)*(18 + 7*an)*np.power(k2,6))*np.power(np.abs(k1 - k2),an)) +
       49*(-2 + an)*an*(4 + an)*(6 + an)*np.power(b2k2,2)*np.power(k1,4)*
        (np.power(k1 + k2,2 + an) - np.power(np.abs(k1 - k2),2 + an)) -
       392*(-2 + an)*(6 + an)*b2k2*bG2k2*np.power(k1,2)*
        (-(np.power(k1 + k2,2 + an)*(np.power(k1,2) - (2 + an)*k1*k2 + np.power(k2,2))) +
          (np.power(k1,2) + (2 + an)*k1*k2 + np.power(k2,2))*np.power(np.abs(k1 - k2),2 + an)) +
       1568*np.power(bG2k2,2)*(np.power(k1 + k2,2 + an)*
           (3*np.power(k1,4) - 3*(2 + an)*np.power(k1,3)*k2 +
             (6 + an*(4 + an))*np.power(k1,2)*np.power(k2,2) - 3*(2 + an)*k1*np.power(k2,3) +
             3*np.power(k2,4)) - (3*np.power(k1,4) + 3*(2 + an)*np.power(k1,3)*k2 +
             (6 + an*(4 + an))*np.power(k1,2)*np.power(k2,2) + 3*(2 + an)*k1*np.power(k2,3) +
             3*np.power(k2,4))*np.power(np.abs(k1 - k2),2 + an))))/(392.*(-2 + an)*an*(2 + an)*(4 + an)*(6 + an)*np.power(k1,5)*k2)

def T2211_A_s_k2ok1(k1, k2, b1k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)
    eps = k2 / k1

    return (np.power(b1k1,2)*b2k2*np.power(k1,an)*
     (42*b2k2 + (-4*b1k2 + 7*an*(1 + an)*b2k2 - 112*bG2k2)*np.power(eps,2)))/168.

def T2211_A_s_k1ok2(k1, k2, b1k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)
    eps = k1 / k2

    return (Power(b1k1,2)*(6272*Power(bG2k2,2)*Power(k2,an)*(14 + (-4 + an)*(1 + an)*Power(eps,2)) - 
       10976*b2k2*bG2k2*Power(k2,an)*(10 + (-2 + an)*(1 + an)*Power(eps,2)) + 
       6860*Power(b2k2,2)*Power(k2,an)*(6 + an*(1 + an)*Power(eps,2)) + 
       (Power(b1k2,2)*Power(k2,an)*(13720 + 28*(1138 + 7*an*(-115 + 21*an))*Power(eps,2) + 
            (-4 + an)*(4026 + an*(4327 + 7*an*(-314 + 35*an)))*Power(eps,4)))/Power(eps,2) + 
       b1k2*(224*bG2k2*Power(k2,an)*
           (-462 + 98*an + (-4 + an)*(1 + an)*(-47 + 7*an)*Power(eps,2)) + 
          392*b2k2*Power(k2,an)*(130 - 70*an - 
             (-2 + an)*(1 + an)*(-27 + 7*an)*Power(eps,2)))))/164640.

def T2211_A_squeezed(k1, b1, b2, bG2, an):
    k1 = np.asarray(k1, dtype=complex)

    return (np.power(2,-3 + an)*np.power(b1,2)*((1016 + 7*an*(-10 + 7*an))*np.power(b1,2) -
       28*b1*((6 + an)*(-12 + 7*an)*b2 + 4*(38 - 7*an)*bG2) +
       196*((4 + an)*(6 + an)*np.power(b2,2) - 8*(6 + an)*b2*bG2 + 32*np.power(bG2,2))) * np.power(k1,an))/(49.*(2 + an)*(4 + an)*(6 + an))

def T2211_X_kernel(k1, k2, b1k1, b2k1, bG2k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)

    return(b1k1*b1k2 * (-7*(-2 + an)*(6 + an)*b1k1*b2k2*np.power(k1,2)*
        (np.power(k1 + k2,1 + an)*(-((8 + 7*an)*np.power(k1,3)) +
             (1 + an)*(8 + 7*an)*np.power(k1,2)*k2 - 20*(1 + an)*k1*np.power(k2,2) +
             20*np.power(k2,3)) + (k1 - k2)*
           ((8 + 7*an)*np.power(k1,3) + (1 + an)*(8 + 7*an)*np.power(k1,2)*k2 +
             20*(1 + an)*k1*np.power(k2,2) + 20*np.power(k2,3))*np.power(np.abs(k1 - k2),an)) -
       7*(-2 + an)*(6 + an)*b1k2*b2k1*np.power(k2,2)*
        (np.power(k1 + k2,1 + an)*(20*np.power(k1,3) - 20*(1 + an)*np.power(k1,2)*k2 +
             (1 + an)*(8 + 7*an)*k1*np.power(k2,2) - (8 + 7*an)*np.power(k2,3)) -
          (k1 - k2)*(20*np.power(k1,3) + 20*(1 + an)*np.power(k1,2)*k2 +
             (1 + an)*(8 + 7*an)*k1*np.power(k2,2) + (8 + 7*an)*np.power(k2,3))*
           np.power(np.abs(k1 - k2),an)) +
       28*b1k1*bG2k2*(np.power(k1 + k2,1 + an)*
           (3*(2 + 7*an)*np.power(k1,5) - 3*(1 + an)*(2 + 7*an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,3)*np.power(k2,2) -
             (1 + an)*(42 + 47*an)*np.power(k1,2)*np.power(k2,3) +
             120*(1 + an)*k1*np.power(k2,4) - 120*np.power(k2,5)) -
          (k1 - k2)*(3*(2 + 7*an)*np.power(k1,5) + 3*(1 + an)*(2 + 7*an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,3)*np.power(k2,2) +
             (1 + an)*(42 + 47*an)*np.power(k1,2)*np.power(k2,3) +
             120*(1 + an)*k1*np.power(k2,4) + 120*np.power(k2,5))*np.power(np.abs(k1 - k2),an)) +
       b1k2*bG2k1*(-28*np.power(k1 + k2,1 + an)*
           (120*np.power(k1,5) - 120*(1 + an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + 47*an)*np.power(k1,3)*np.power(k2,2) -
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,2)*np.power(k2,3) +
             3*(1 + an)*(2 + 7*an)*k1*np.power(k2,4) - 3*(2 + 7*an)*np.power(k2,5)) +
          28*(k1 - k2)*(120*np.power(k1,5) + 120*(1 + an)*np.power(k1,4)*k2 +
             (1 + an)*(42 + 47*an)*np.power(k1,3)*np.power(k2,2) +
             (1 + an)*(42 + an*(9 + 7*an))*np.power(k1,2)*np.power(k2,3) +
             3*(1 + an)*(2 + 7*an)*k1*np.power(k2,4) + 3*(2 + 7*an)*np.power(k2,5))*
           np.power(np.abs(k1 - k2),an)) +
       b1k1*b1k2*(-(np.power(k1 + k2,an)*
             (60*(2 + 7*an)*np.power(k1,6) - 60*an*(2 + 7*an)*np.power(k1,5)*k2 +
               (-120 + an*(676 + an*(250 + 189*an)))*np.power(k1,4)*np.power(k2,2) -
               an*(-80 + an*(816 + 7*an*(30 + 7*an)))*np.power(k1,3)*np.power(k2,3) +
               (-120 + an*(676 + an*(250 + 189*an)))*np.power(k1,2)*np.power(k2,4) -
               60*an*(2 + 7*an)*k1*np.power(k2,5) + 60*(2 + 7*an)*np.power(k2,6))) +
          (60*(2 + 7*an)*np.power(k1,6) + 60*an*(2 + 7*an)*np.power(k1,5)*k2 +
             (-120 + an*(676 + an*(250 + 189*an)))*np.power(k1,4)*np.power(k2,2) +
             an*(-80 + an*(816 + 7*an*(30 + 7*an)))*np.power(k1,3)*np.power(k2,3) +
             (-120 + an*(676 + an*(250 + 189*an)))*np.power(k1,2)*np.power(k2,4) +
             60*an*(2 + 7*an)*k1*np.power(k2,5) + 60*(2 + 7*an)*np.power(k2,6))*
           np.power(np.abs(k1 - k2),an)) +
       49*(-2 + an)*an*(4 + an)*(6 + an)*b2k1*b2k2*np.power(k1,2)*np.power(k2,2)*
        (np.power(k1 + k2,2 + an) - np.power(np.abs(k1 - k2),2 + an)) -
       196*(-2 + an)*(6 + an)*b2k2*bG2k1*np.power(k1,2)*
        (-(np.power(k1 + k2,2 + an)*(np.power(k1,2) - (2 + an)*k1*k2 + np.power(k2,2))) +
          (np.power(k1,2) + (2 + an)*k1*k2 + np.power(k2,2))*np.power(np.abs(k1 - k2),2 + an)) -
       196*(-2 + an)*(6 + an)*b2k1*bG2k2*np.power(k2,2)*
        (-(np.power(k1 + k2,2 + an)*(np.power(k1,2) - (2 + an)*k1*k2 + np.power(k2,2))) +
          (np.power(k1,2) + (2 + an)*k1*k2 + np.power(k2,2))*np.power(np.abs(k1 - k2),2 + an)) +
       1568*bG2k1*bG2k2*(np.power(k1 + k2,2 + an)*
           (3*np.power(k1,4) - 3*(2 + an)*np.power(k1,3)*k2 +
             (6 + an*(4 + an))*np.power(k1,2)*np.power(k2,2) - 3*(2 + an)*k1*np.power(k2,3) +
             3*np.power(k2,4)) - (3*np.power(k1,4) + 3*(2 + an)*np.power(k1,3)*k2 +
             (6 + an*(4 + an))*np.power(k1,2)*np.power(k2,2) + 3*(2 + an)*k1*np.power(k2,3) +
             3*np.power(k2,4))*np.power(np.abs(k1 - k2),2 + an))))/(392.*(-2 + an)*an*(2 + an)*(4 + an)*(6 + an)*np.power(k1,3)*np.power(k2,3))

def T2211_X_s_k2ok1(k1, k2, b1k1, b2k1, bG2k1, b1k2, b2k2, bG2k2, an):
    k1 = np.asarray(k1, dtype=complex)
    k2 = np.asarray(k2, dtype=complex)
    eps = k2/k1

    return (b1k1*b1k2*(10*(-8 + 21*an)*b1k1*b1k2*Power(k1,an)*Power(eps,2) - 
       7*(-2 + an)*(1 + an)*(-27 + 7*an)*b1k1*b2k2*Power(k1,an)*Power(eps,2) + 
       56*(-33 + 7*an)*b1k1*bG2k2*Power(k1,an)*Power(eps,2) + 
       Power(k1,an)*(-70*b2k2*((-13 + 7*an)*b1k1 - 21*b2k1 + 28*bG2k1) - 
          70*b1k2*(b2k1 + 4*bG2k1)*Power(eps,2) + 
          49*(1 + an)*b2k2*(5*an*b2k1 + 8*bG2k1 - 4*an*bG2k1)*Power(eps,2) + 
          392*(-5*b2k1 + 8*bG2k1)*bG2k2*Power(eps,2))))/5880.

def T2211_X_squeezed(k1, b1, b2, bG2, an):
    k1 = np.asarray(k1, dtype=complex)

    return (Power(2,-3 + an)*Power(b1,2)*((1016 + 7*an*(-10 + 7*an))*Power(b1,2) - 
       28*(6 + an)*(-12 + 7*an)*b1*b2 + 196*(4 + an)*(6 + an)*Power(b2,2) + 
       112*(-38 + 7*an)*b1*bG2 - 1568*(6 + an)*b2*bG2 + 6272*Power(bG2,2))*
     Power(k1,an))/(49.*(2 + an)*(4 + an)*(6 + an))


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

    return (np.power(k1 + k2, an + 2) - np.power(np.abs(k1 - k2), an + 2)) / (
        2 * k1 * k2 * (an + 2)
    )
