# RANSAC

---

### How to Run

###### Open `RANSAC.py` and run it directly, and then input according to the prompts to draw the result

---

> RANSAC algorithm: According to the iterative rounds of iters, two points are **randomly** generated in each round, and the connected line is obtained, and then the `error()` of each point with the line is calculated, and the number of points with the error **less than** `eps` is calculated (Now, These points can be considered as the inliers of the straight line). Continuously update the maximum number of interior points and the coordinates of the corresponding interior points. When the ratio of the number of interior points is greater than `inliers_threshold`, it will exit the iteration directly, and the result has met the requirements; otherwise, the best result is the final result.

```python
def RANSAC(points_set, iters=1000, eps=0.2, inliers_threshold=0.8):
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
```

---

### Use `leastsq()` for least squares method, where `error` is the error function

> ```python
> def error(Parm, x, y):
>    '''
>    :param Parm: [a,b] Slope and intercept
>    :param x: Point abscissa
>    :param y: Point ordinate
>    :return: Error between current point and parameter straight line
>    '''
>    return Parm[0] * x + Parm[1] - y
> ```

```python
    p0 = np.array([best_inliers_x[0], best_inliers_y[0]])
    best_inliers_x = np.array(best_inliers_x)
    best_inliers_y = np.array(best_inliers_y)
    best_a, best_b = leastsq(error, p0, args=(best_inliers_x, best_inliers_y))[0]
```

---

### Draw scatter plots and evaluate straight lines

```python
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
```

---

### RANSAC linear result drawing of randomly generated points (rather than specified points) for testing convenience

```python
points_set = []
    for _ in range(points_cnt):
        points_set.append([random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])])

    a, b = RANSAC(points_set, iters=1e5, eps=0.1)
    X_s, Y_s = np.array(points_set).T
    draw(X_s, Y_s, a, b)
```
