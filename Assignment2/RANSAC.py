import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def error(Parm, x, y):
    '''
    :param Parm: [a,b] Slope and intercept
    :param x: Point abscissa
    :param y: Point ordinate
    :return: Error between current point and parameter straight line
    '''
    return Parm[0] * x + Parm[1] - y


def RANSAC(points_set, iters=1000, eps=0.2, inliers_threshold=0.8):
    '''
    :param points_set: points set
    :param iters: the count of iterations
    :param eps: Allowable error
    :param inliers_threshold: Inlier ratio threshold
    :return: Linear RANSAC optimal parameters
    '''

    points_cnt = len(points_set)
    best_inliers_x = []
    best_inliers_y = []
    for _ in range(int(iters)):

        sample_index = random.sample(range(points_cnt), 2)
        [x1, y1], [x2, y2] = points_set[sample_index[0]], points_set[sample_index[1]]

        now_a = (y2 - y1) / (x2 - x1)
        now_b = y1 - now_a * x1

        inliers_x = []
        inliers_y = []
        for [now_x, now_y] in points_set:
            if abs(error([now_a, now_b], now_x, now_y)) < eps:
                inliers_x.append(now_x)
                inliers_y.append(now_y)

        if len(inliers_x) > len(best_inliers_x):
            best_inliers_x = inliers_x
            best_inliers_y = inliers_y

        if len(best_inliers_x) >= int(points_cnt * inliers_threshold):
            break

    p0 = np.array([best_inliers_x[0], best_inliers_y[0]])
    best_inliers_x = np.array(best_inliers_x)
    best_inliers_y = np.array(best_inliers_y)
    best_a, best_b = leastsq(error, p0, args=(best_inliers_x, best_inliers_y))[0]

    return best_a, best_b


def draw(X_s, Y_s, a, b):
    '''

    :param X_s: Abscissa set
    :param Y_s: Ordinate set
    :param a:  Slope
    :param b:  Intercept
    :return: a figure
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title("RANSAC")
    ax1.scatter(X_s, Y_s)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    Y = a * X_s + b

    ax1.plot(X_s, Y)
    minX, maxX = min(X_s), max(X_s)
    minY, maxY = min(Y_s), max(Y_s)
    plt.text(minX + (maxX - minX) / 1.5, minY + (maxY - minY) / 5, "best_a=" + str(a) + "\nbest_b=" + str(b),
             fontdict={"size": 10, "color": "b", "alpha": 0.5}, bbox={'facecolor': '#74C476',
                                                                      'edgecolor': 'b',
                                                                      'alpha': 0.3,
                                                                      'pad': 8,
                                                                      }
             )
    plt.show()


def randomTest(points_cnt, x_range, y_range):
    '''

    :param points_cnt:  the number of sactter
    :param x_range: [x_min,x_max]
    :param y_range: [y_min,y_max]
    :return:
    '''
    points_set = []
    for _ in range(points_cnt):
        points_set.append([random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])])

    a, b = RANSAC(points_set, iters=1e5, eps=0.1)
    X_s, Y_s = np.array(points_set).T
    draw(X_s, Y_s, a, b)


if __name__ == "__main__":
    is_random = input("Random point [1] or Designated point [2]? >>>")

    if is_random == "1":
        n = int(input("points nums : "))
        if(n<=0):
            print("Invalid Input!")
            exit(-1)
        x_min, x_max = float(input("x_min=")), float(input("x_max="))
        y_min, y_max = float(input("y_min=")), float(input("y_max="))
        if x_min >= x_max or y_min >= y_max:
            print("Invalid Input!")
            exit(-1)
        randomTest(n, [x_min, x_max], [y_min, y_max])
    elif is_random == "2":
        n = int(input("points nums : "))
        if (n <= 0):
            print("Invalid Input!")
            exit(-1)
        points_set = []
        for i in range(n):
            x, y = float(input("x" + str(i) + "=")), float(input("y" + str(i) + "="))
            points_set.append([x, y])

        a, b = RANSAC(points_set)
        X_s, Y_s = np.array(points_set).T
        draw(X_s, Y_s, a, b)
    else:
        print("Invalid Input!")
        exit(-1)
