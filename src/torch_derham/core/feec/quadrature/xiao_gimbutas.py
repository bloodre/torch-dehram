"""Xiao-Gimbutas quadrature rules for triangles and tetrahedra.

Modern high-efficiency symmetric quadrature rules with minimal number of points
for a given degree of exactness. These rules are significantly more efficient
than classical Dunavant or Keast rules.

Sources:
    - Xiao, H. and Gimbutas, Z., "A numerical algorithm for the construction
      of efficient quadrature rules in two and higher dimensions",
      Computers & Mathematics with Applications, Vol. 59, No. 2, 2010,
      pp. 663-676.
      https://doi.org/10.1016/j.camwa.2009.10.027
      
    - Data retrieved from quadraturerules.org:
      https://quadraturerules.org/Q000002/
      
    - Implementation reference: modepy library
      https://github.com/inducer/modepy/blob/main/modepy/quadrature/xiao_gimbutas.py

Notes:
    These rules are optimal or near-optimal in terms of the number of quadrature
    points required for a given polynomial degree. They use numerical optimization
    to minimize point count while maintaining symmetry and positive weights.
    
Reference simplex:
    - Triangle: vertices at (1,0,0), (0,1,0), (0,0,1) in barycentric coords.
    - Tetrahedron: vertices at (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1).
"""

from __future__ import annotations


def xiao_gimbutas_triangle_degree_1() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 1-point rule for triangles.
    
    Exact for polynomials of degree 1.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂]].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.333333333333333, 0.333333333333333, 0.333333333333333],
    ]
    weights = [0.5]
    return points, weights


def xiao_gimbutas_triangle_degree_2() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 3-point rule for triangles.
    
    Exact for polynomials of degree 2.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.666666666666667, 0.166666666666667, 0.166666666666667],
        [0.166666666666667, 0.166666666666667, 0.666666666666667],
        [0.166666666666667, 0.666666666666667, 0.166666666666667],
    ]
    weights = [0.166666666666667, 0.166666666666667, 0.166666666666667]
    return points, weights


def xiao_gimbutas_triangle_degree_3() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 6-point rule for triangles.
    
    Exact for polynomials of degree 3.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.108103018168070, 0.445948490915965, 0.445948490915965],
        [0.816847572980458, 0.091576213509771, 0.091576213509771],
        [0.445948490915965, 0.445948490915965, 0.108103018168070],
        [0.091576213509771, 0.091576213509771, 0.816847572980458],
        [0.445948490915965, 0.108103018168070, 0.445948490915965],
        [0.091576213509771, 0.816847572980458, 0.091576213509771],
    ]
    weights = [
        0.111690794839005,
        0.054975871827661,
        0.111690794839005,
        0.054975871827661,
        0.111690794839005,
        0.054975871827661,
    ]
    return points, weights


def xiao_gimbutas_triangle_degree_4() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 6-point rule for triangles.
    
    Exact for polynomials of degree 4.
    Same points as degree 3 (rule is exact to degree 4).
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    return xiao_gimbutas_triangle_degree_3()


def xiao_gimbutas_triangle_degree_5() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 7-point rule for triangles.
    
    Exact for polynomials of degree 5.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.333333333333333, 0.333333333333333, 0.333333333333333],
        [0.797426985353087, 0.101286507323456, 0.101286507323456],
        [0.059715871789770, 0.470142064105115, 0.470142064105115],
        [0.101286507323456, 0.101286507323456, 0.797426985353087],
        [0.470142064105115, 0.470142064105115, 0.059715871789770],
        [0.101286507323456, 0.797426985353087, 0.101286507323456],
        [0.470142064105115, 0.059715871789770, 0.470142064105115],
    ]
    weights = [
        0.1125,
        0.062969590272413,
        0.066197076394253,
        0.062969590272413,
        0.066197076394253,
        0.062969590272413,
        0.066197076394253,
    ]
    return points, weights


def xiao_gimbutas_tetrahedron_degree_1() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 1-point rule for tetrahedra.
    
    Exact for polynomials of degree 1.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃]].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.25, 0.25, 0.25, 0.25],
    ]
    weights = [0.166666666666667]
    return points, weights


def xiao_gimbutas_tetrahedron_degree_2() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 4-point rule for tetrahedra.
    
    Exact for polynomials of degree 2.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.123666800328458, 0.821572540967620, 0.039933048641498, 0.014827610062423],
        [0.457461587085596, 0.155933120499186, 0.381765356069347, 0.004839936345872],
        [0.365314518814635, 0.180029693510365, 0.006923235573627, 0.447732552101373],
        [0.000375515028729, 0.216076429184848, 0.430701707077836, 0.352846348708587],
    ]
    weights = [
        0.016934258079135,
        0.046462961277761,
        0.050086823227827,
        0.053182624081944,
    ]
    return points, weights


def xiao_gimbutas_tetrahedron_degree_3() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 6-point rule for tetrahedra.
    
    Exact for polynomials of degree 3.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.641429791495696, 0.162001491698525, 0.183850350492098, 0.012718366313681],
        [0.345444155719731, 0.010905212211189, 0.281523802123546, 0.362126829945534],
        [0.439858947649275, 0.190117002439284, 0.011403329444557, 0.358620720466884],
        [0.037871631782357, 0.170816925164989, 0.152818143090927, 0.638493299961727],
        [0.124804862165247, 0.158685163227441, 0.585662805655216, 0.130847168952096],
        [0.141482751969505, 0.571226052149115, 0.146918390087170, 0.140372805794211],
    ]
    weights = [
        0.020387000459558,
        0.021344402118458,
        0.022094671190741,
        0.023437401610067,
        0.037402527819593,
        0.042000663186750,
    ]
    return points, weights


def xiao_gimbutas_tetrahedron_degree_4() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 11-point rule for tetrahedra.
    
    Exact for polynomials of degree 4.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.174694058697231, 0.040490506727590, 0.013560701879803, 0.771254732695376],
        [0.081404918402859, 0.752508507009655, 0.068099370938207, 0.097987203649279],
        [0.741228882093623, 0.067223294893383, 0.035183929773599, 0.156363893239395],
        [0.053341239535745, 0.419266313879513, 0.047781435559087, 0.479611011025656],
        [0.432953490481356, 0.450765876091277, 0.059456616299434, 0.056824017127934],
        [0.538007203916186, 0.129411373788910, 0.330190414837465, 0.002391007457439],
        [0.008991260093336, 0.121541991333928, 0.306493988429690, 0.562972760143046],
        [0.106604172561994, 0.097204644587583, 0.684390415453040, 0.111800767397383],
        [0.329232959742647, 0.029569495206480, 0.317903560213395, 0.323293984837479],
        [0.103844116410993, 0.432710239047769, 0.353823239209297, 0.109622405331941],
        [0.304448402434497, 0.240276664928073, 0.126801725915392, 0.328473206722038],
    ]
    weights = [
        0.006541848487473,
        0.009212228192281,
        0.009232299812094,
        0.009988863857593,
        0.011578327655606,
        0.012693785874260,
        0.013237780011338,
        0.017744672525817,
        0.018372387138083,
        0.025829352693577,
        0.032234801674391,
    ]
    return points, weights


def xiao_gimbutas_tetrahedron_degree_5() -> tuple[list[list[float]], list[float]]:
    """Xiao-Gimbutas 14-point rule for tetrahedra.
    
    Exact for polynomials of degree 5.
    
    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.454496295874350, 0.454496295874350, 0.045503704125650, 0.045503704125650],
        [0.045503704125650, 0.454496295874350, 0.454496295874350, 0.045503704125650],
        [0.045503704125650, 0.454496295874350, 0.045503704125650, 0.454496295874350],
        [0.454496295874350, 0.045503704125650, 0.454496295874350, 0.045503704125650],
        [0.454496295874350, 0.045503704125650, 0.045503704125650, 0.454496295874351],
        [0.045503704125650, 0.045503704125650, 0.454496295874350, 0.454496295874350],
        [0.092735250310891, 0.721794249067326, 0.092735250310891, 0.092735250310891],
        [0.721794249067326, 0.092735250310891, 0.092735250310891, 0.092735250310891],
        [0.092735250310891, 0.092735250310891, 0.092735250310891, 0.721794249067326],
        [0.092735250310891, 0.092735250310891, 0.721794249067326, 0.092735250310891],
        [0.310885919263301, 0.067342242210098, 0.310885919263301, 0.310885919263300],
        [0.067342242210098, 0.310885919263301, 0.310885919263301, 0.310885919263301],
        [0.310885919263301, 0.310885919263301, 0.310885919263301, 0.067342242210098],
        [0.310885919263301, 0.310885919263301, 0.067342242210098, 0.310885919263301],
    ]
    weights = [
        0.007091003462847,
        0.007091003462847,
        0.007091003462847,
        0.007091003462847,
        0.007091003462847,
        0.007091003462847,
        0.012248840518606,
        0.012248840518606,
        0.012248840518606,
        0.012248840518606,
        0.018781320530030,
        0.018781320530030,
        0.018781320530030,
        0.018781320530027,
    ]
    return points, weights
