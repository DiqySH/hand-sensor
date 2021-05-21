from math import sqrt


def get_average_knuckle_distance(points):
    # Check the average distance between all the knuckles of
    # the hand's index, middle, ring and pinky finger
    # Knuckles start at point 5 and go to point 20
    total_distance = 0
    for i in range(6, len(points) - 1):
        total_distance += distance_between(points[i].x,
                                           points[i].y, points[i - 1].x, points[i - 1].y)
    return total_distance / 16

def get_palm_coordinate(p):
    # get the coordinate average of points 0, 1, 5, 9, 13, 17
    # to get an accurate estimation of the location of the palm
    x = (p[0].x + p[1].x + p[5].x + p[9].x + p[13].x + p[17].x) / 6
    y = (p[0].y + p[1].y + p[5].y + p[9].y + p[13].y + p[17].y) / 6
    return (x, y)
    

def get_root_distance(root1, root2):
    return distance_between(root1.x, root1.y, root2.x, root2.y)


def distance_between(x1, y1, x2, y2):
    # Return the distance between 2 points
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def map_range(value, start1, stop1, start2, stop2):
    # Map a value from one range to another
    return (value - start1) / (stop1 - start1) * (stop2 - start2) + start2
