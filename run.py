import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Helper functions

def compute_k(F):
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    U, _, _ = np.linalg.svd(F)
    t = U[:, 2]
    sk_matrix = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    P2 = np.hstack((np.dot(sk_matrix, F) + t[:, None], t[:, None]))

    # Compute K1
    K1 = P1[:, :3]

    # Compute K2
    M = P2[:, :3]
    lam = 1 / np.linalg.norm(M[2, :]) if np.linalg.det(M) > 0 else -1 / np.linalg.norm(M[2, :])
    M = M * lam

    _, K2 = np.linalg.qr(np.linalg.inv(M))
    K2 = np.linalg.inv(K2)

    # Ensure diagonals are positive
    for i in range(3):
        if K1[i, i] < 0:
            K1[:, i] = -K1[:, i]
    for i in range(3):
        if K2[i, i] < 0:
            K2[:, i] = -K2[:, i]

    return K1, K2

def triangulate_points(x1, x2, P1, P2):
    X = []
    for i in range(len(x1)):
        A = np.array([
            x1[i, 0] * P1[2, :] - P1[0, :],
            x1[i, 1] * P1[2, :] - P1[1, :],
            x2[i, 0] * P2[2, :] - P2[0, :],
            x2[i, 1] * P2[2, :] - P2[1, :]
        ])
        _, _, V = np.linalg.svd(A)
        X.append(V[-1, :] / V[-1, -1])
    return np.array(X).T

def plot_epipolar_lines(lines, points, title):
    plt.figure()
    plt.title(title)
    for i in range(len(points)):
        l = lines[:, i]
        hline(l)
        plt.plot(points[i, 0], points[i, 1], 'ro')
    plt.show()

def hline(l):
    if abs(l[0]) < abs(l[1]):
        xlim = plt.gca().get_xlim()
        x1 = np.cross(l, [1, 0, -xlim[0]])
        x2 = np.cross(l, [1, 0, -xlim[1]])
    else:
        ylim = plt.gca().get_ylim()
        x1 = np.cross(l, [0, 1, -ylim[0]])
        x2 = np.cross(l, [0, 1, -ylim[1]])
    x1 = x1 / x1[2]
    x2 = x2 / x2[2]
    plt.plot([x1[0], x2[0]], [x1[1], x2[1]])

def error(E, p1, p2):
    l1_error = np.dot(E.T, p1)
    l2_error = np.dot(E, p2)
    a = np.sum((np.dot(p2.T, l2_error))**2)
    b = l2_error[0, :]**2 + l2_error[1, :]**2 + l1_error[0, :]**2 + l1_error[1, :]**2
    return np.sum(a/b)

def compute_F(input_1, input_2):
    input_1T, T_1 = homography_condition(input_1)
    input_2T, T_2 = homography_condition(input_2)
    F_t = fundamental_matrix(input_1T, input_2T)
    F = homography_decondition(F_t, T_1, T_2)
    F = enforce_singularity(F)
    return F

def fundamental_matrix(i_1, i_2):
    design_mat = []
    for i in range(len(i_1)):
        temp_mat = np.array([i_1[0,i]*i_2[0,i], i_1[1,i]*i_2[0,i], i_2[0,i], 
                             i_1[0,i]*i_2[1,i], i_1[1,i]*i_2[1,i], i_2[1,i], 
                             i_1[0,i], i_1[1,i], 1])
        design_mat.append(temp_mat)
    design_mat = np.array(design_mat)
    _, _, V = np.linalg.svd(design_mat)
    return V[-1].reshape(3, 3)

def homography_condition(input_2d):
    mean_2d = np.mean(input_2d, axis=0)
    centered_2d = input_2d - mean_2d
    scale_2d = np.mean(np.abs(centered_2d), axis=0)
    T_2d = np.dot(np.diag([1/scale_2d[0], 1/scale_2d[1], 1]), np.array([[1, 0, -mean_2d[0]], [0, 1, -mean_2d[1]], [0, 0, 1]]))
    input_2d_T = np.dot(T_2d, np.hstack((input_2d, np.ones((len(input_2d), 1)))).T)
    return input_2d_T, T_2d

def homography_decondition(f, T_1, T_2):
    return np.dot(np.dot(T_2.T, f), T_1)

def enforce_singularity(F):
    U, D, V = np.linalg.svd(F)
    if np.linalg.det(F) != 0:
        D[-1] = 0
        F_p = np.dot(U, np.dot(np.diag(D), V))
    else:
        F_p = F
    return F_p

# Main script

# Load data
data = np.loadtxt('calib_points.dat')
x11 = data[:2, :].T
x22 = data[2:4, :].T
X_E, Y_E, Z_E = data[4, :], data[5, :], data[6, :]

# Compute F and K matrices
F = compute_F(x11, x22)
K1, K2 = compute_k(F)

# Calibrated points
x11c = np.dot(np.linalg.inv(K1), np.hstack((x11, np.ones((12,1)))).T)[:2, :].T
x22c = np.dot(np.linalg.inv(K2), np.hstack((x22, np.ones((12,1)))).T)[:2, :].T

print("K1:\n", K1)
print("K2:\n", K2)

# Compute essential matrix E
E = compute_F(x11c, x22c)

# Enforce singularity
U, D, V = np.linalg.svd(E)
if np.linalg.det(E) != 0:
    D[-1] = 0
    E_p = np.dot(U, np.dot(np.diag(D), V))
else:
    E_p = E

print("Essential Matrix:\n", E_p)

# Fourfold ambiguity
skew_symmetric_matrix = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
rotation1 = np.dot(U, np.dot(skew_symmetric_matrix, V))
rotation2 = np.dot(U, np.dot(skew_symmetric_matrix.T, V))
translation1 = U[:, 2]
translation2 = -U[:, 2]

projection1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
projection2a = np.dot(K2, np.hstack((rotation1, translation1[:, None])))
projection2b = np.dot(K2, np.hstack((rotation1, translation2[:, None])))
projection2c = np.dot(K2, np.hstack((rotation2, translation1[:, None])))
projection2d = np.dot(K2, np.hstack((rotation2, translation2[:, None])))

object_points_a = triangulate_points(x11c, x22c, projection1, projection2a)
object_points_b = triangulate_points(x11c, x22c, projection1, projection2b)
object_points_c = triangulate_points(x11c, x22c, projection1, projection2c)
object_points_d = triangulate_points(x11c, x22c, projection1, projection2d)

constraint_a = np.sum(object_points_a[2, :] > 0)
constraint_b = np.sum(object_points_b[2, :] > 0)
constraint_c = np.sum(object_points_c[2, :] > 0)
constraint_d = np.sum(object_points_d[2, :] > 0)

constraint_counts = [constraint_a, constraint_b, constraint_c, constraint_d]
max_index = np.argmax(constraint_counts)

selected_projection1 = projection1
if max_index == 0:
    selected_projection2 = projection2a
    selected_object_points = object_points_a
elif max_index == 1:
    selected_projection2 = projection2b
    selected_object_points = object_points_b
elif max_index == 2:
    selected_projection2 = projection2c
    selected_object_points = object_points_c
else:
    selected_projection2 = projection2d
    selected_object_points = object_points_d

print("Geometrically plausible solution:")
print("Projection1:\n", selected_projection1)
print("Projection2:\n", selected_projection2)

# Visualize epipolar lines
line_1 = np.dot(E_p.T, np.hstack((x22c, np.ones((12,1)))).T)
line_2 = np.dot(E_p, np.hstack((x11c, np.ones((12,1)))).T)

plot_epipolar_lines(line_2, x22c, "Epipolar lines for Non-Optimized Essential Matrix (2nd camera)")
plot_epipolar_lines(line_1, x11c, "Epipolar lines for Non-Optimized Essential Matrix (1st camera)")

# Compute geometric error
geoerror = error(E_p, np.hstack((x11c, np.ones((12,1)))).T, np.hstack((x22c, np.ones((12,1)))).T)
print("Geometric Error with Non-Optimized Parameters:", geoerror)

# Levenberg-Marquart optimization
initial_guess = E_p

# Define the objective function
def objective_function(initial_guess):
    return error(initial_guess, np.hstack((x11c, np.ones((12,1)))).T, np.hstack((x22c, np.ones((12,1)))).T)

# Perform optimization
result = least_squares(objective_function, initial_guess, method='lm', max_nfev=1000, xtol=1e-6)
E_p_optimized = result.x.reshape((3, 3))

print("Optimized Essential Matrix:\n", E_p_optimized)

# Recalculate geometric error
optimized_error = error(E_p_optimized, np.hstack((x11c, np.ones((12,1)))).T, np.hstack((x22c, np.ones((12,1)))).T)
print("Geometric Error with Optimized Parameters:", optimized_error)
