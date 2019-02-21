def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    '''
    Determine the intersection of two lines given only the endpoints
    :return: Coordinates of the lines, if any, in tuple form. It also returns t and u in tuple form.
    if both t and u are between 0 and 1.0, then the intersection exists between the two line segments.
    '''

    Px_a = (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)
    Px_b = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    print Px_a
    print Px_b
    Px = Px_a/Px_b

    Py_a = (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)
    Py_b = Px_b
    Py = Py_a/Py_b

    t_u = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
    t_d = (x1-x2)*(y3-y4)-(y1-y2)*(x3*x4)

    u_u = (x1-x2)*(y1-y3)-(y1-y2)*(x1-x3)
    u_d = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)

    u = -1*(u_u/u_d)
    t = t_u/t_d

    return (Px, Py), (t,u)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


print line_intersection(((0,0), (2,4)), ((4,0), (2,2)))

res, b = intersection(0.,0.,2.,4.,4.,0.,2.,2.)
print res
print b