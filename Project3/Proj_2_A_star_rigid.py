import numpy as np
import cv2
import math


# To check if the new point lies in the circle obstacle
def check_obstacle_circle(point, dimension, clearance):
    increase = dimension + clearance
    center = [190, 20]
    point_x = point[0]
    point_y = point[1]
    dist = np.sqrt((point_x - center[0]) ** 2 + (point_y - center[1]) ** 2)
    if dist <= 15 + increase:
        return True
    else:
        return False


# To check if the new point lies in the ellipse Obstacle
def check_obstacle_ellipse(point, dimension, clearance):
    increase = dimension + clearance
    center = [140, 30]
    rx = 15 + increase
    ry = 6 + increase
    point_x = point[0]
    point_y = point[1]
    dist = (((point_x - center[0]) ** 2) / (rx ** 2)) + (((point_y - center[1]) ** 2) / (ry ** 2))
    if dist <= 1:
        return True
    else:
        return False


# To check if the new point lies in the Square Obstacle
def check_obstacle_square(point, dimension, clearance):
    dist_check = dimension + clearance
    p1 = [50, 37.5]
    p2 = [100, 37.5]
    p3 = [100, 82.5]
    p4 = [50, 82.5]

    point_x = point[0]
    point_y = point[1]

    line1 = (point_y + 1) - (p1[1] - dist_check)
    line2 = point_x - (p2[0] + dist_check)
    line3 = point_y - (p3[1] + dist_check)
    line4 = point_x - (p4[0] - dist_check)

    flag1 = False
    flag2 = False

    if line1 >= 0 and line3 <= 0:
        flag1 = True
    if line2 <= 0 and line4 >= 0:
        flag2 = True

    if flag1 and flag2 is True:
        return True
    else:
        return False


# To calculate the new intersection point of the lines after extension
def intersect_point(c1, c2):
    det = abs(c1[0] - c2[0])

    x_inter, y_inter = None, None
    if det is not 0:
        x_inter = int(round(abs((c1[1] - c2[1])) / det))
        y_inter = int(round(abs(((c1[0] * c2[1]) - (c2[0] * c1[1]))) / det))

    return [x_inter, y_inter]


# Output the new coordinates of the polygon obstacle
def new_poly_points(dimension, clearance):
    increase = dimension + clearance
    p1 = [125, 94]
    p2 = [163, 98]
    p3 = [170, 60]
    p4 = [193, 98]
    p5 = [173, 135]
    p6 = [150, 135]

    coff1 = np.array(np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1))
    coff2 = np.array(np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1))
    coff3 = np.array(np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1))
    coff4 = np.array(np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1))
    coff5 = np.array(np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1))
    coff6 = np.array(np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1))

    if increase < 1:
        return p1, p2, p3, p4, p5, p6
    else:
        # change the intercept formed by the lines
        coff1[1] = coff1[1] - (increase / (math.sin(1.57 - math.atan(coff1[0]))))
        coff2[1] = coff2[1] - (increase / (math.sin(1.57 - math.atan(coff2[0]))))
        coff3[1] = coff3[1] - (increase / (math.sin(1.57 - math.atan(coff3[0]))))
        coff4[1] = coff4[1] + (increase / (math.sin(1.57 - math.atan(coff4[0]))))
        coff5[1] = coff5[1] + (increase / (math.sin(1.57 - math.atan(coff5[0]))))
        coff6[1] = coff6[1] + (increase / (math.sin(1.57 - math.atan(coff6[0]))))

        # Keep the slope constant but changing intercept find the intersection point
        p2 = intersect_point(coff1, coff2)
        p3 = intersect_point(coff2, coff3)
        p4 = intersect_point(coff3, coff4)
        p5 = intersect_point(coff4, coff5)
        p6 = intersect_point(coff5, coff6)
        p1 = intersect_point(coff6, coff1)

        return p1, p2, p3, p4, p5, p6


"""I have divided the non convex obstacle shape into sets on convex figures so that it is
easier to see if a point lies inside it or not"""


def check_obstacle_poly(pt, poly_points_updated):
    point_x = pt[0]
    point_y = pt[1]

    p1 = poly_points_updated[0]
    p2 = poly_points_updated[1]
    p3 = poly_points_updated[2]
    p4 = poly_points_updated[3]
    p5 = poly_points_updated[4]
    p6 = poly_points_updated[5]

    coff1 = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    coff2 = np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1)
    coff3 = np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1)
    coff4 = np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1)
    coff5 = np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1)
    coff6 = np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1)

    # To divide the figure into 4 triangles we make temporary lines
    coff_temp26 = np.polyfit([p2[0], p6[0]], [p2[1], p6[1]], 1)
    coff_temp24 = np.polyfit([p2[0], p4[0]], [p2[1], p4[1]], 1)
    coff_temp25 = np.polyfit([p2[0], p5[0]], [p2[1], p5[1]], 1)

    line1 = round(point_y - coff1[0] * point_x - (coff1[1]))
    line2 = round(point_y - coff2[0] * point_x - (coff2[1]))
    line3 = round(point_y - coff3[0] * point_x - (coff3[1]))
    line4 = round(point_y - coff4[0] * point_x - (coff4[1]))
    line5 = round(point_y - coff5[0] * point_x - (coff5[1]))
    line6 = round(point_y - coff6[0] * point_x - (coff6[1]))
    line_temp26 = round(point_y - coff_temp26[0] * point_x - (coff_temp26[1]))
    line_temp24 = round(point_y - coff_temp24[0] * point_x - (coff_temp24[1]))
    line_temp25 = round(point_y - coff_temp25[0] * point_x - (coff_temp25[1]))

    # Different flags for different triangles in the poly shape
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False

    if line1 >= 0 and line_temp26 <= 0 and line6 <= 0:
        flag1 = True
    if line2 >= 0 and line3 >= 0 and line_temp24 <= 0:
        flag2 = True
    if line_temp24 >= 0 and line4 <= 0 and line_temp25 <= 0:
        flag3 = True
    if line_temp25 >= 0 and line5 <= 0 and line_temp26 >= 0:
        flag4 = True

    if flag1 or flag2 or flag3 or flag4 is True:
        return True
    else:
        return False


def check_obstacle_corner(point, dimension, clearance):
    increase = dimension + clearance

    p1 = [increase, increase]
    p2 = [250 - increase, increase]
    p3 = [250 - increase, 150 - increase]
    p4 = [increase, 150 - increase]

    if point[0] <= p1[0] or point[0] >= p2[0]:
        return True
    elif point[1] <= p1[1] or point[1] >= p3[1]:
        return True
    else:
        return False


def call_obstacle_checks(point, dimension, clearance, poly_points_updated):
    if check_obstacle_poly(point, poly_points_updated):
        return True
    elif check_obstacle_square(point, dimension, clearance):
        return True
    elif check_obstacle_ellipse(point, dimension, clearance):
        return True
    elif check_obstacle_circle(point, dimension, clearance):
        return True
    elif check_obstacle_corner(point, dimension, clearance):
        return True
    else:
        return False


# plot the obstacle space


def plot_workspace(points_crown, dimension, clearance):
    increase = dimension + clearance
    img = 255 * np.ones((151, 251, 3), np.uint8)

    # Plot the right part of the crown shape
    cords_crown1 = np.array([points_crown[1], points_crown[2], points_crown[3], points_crown[4], points_crown[5]],
                            dtype=np.int32)
    cords_crown = np.array([[163, 98], [170, 60], [193, 98], [173, 135], [150, 135]],
                           dtype=np.int32)
    cv2.fillConvexPoly(img, cords_crown1, 255)
    cv2.fillConvexPoly(img, cords_crown, 0)

    # Plot the triangle on the left of the crown
    cords_crown_triangle1 = np.array([points_crown[0], points_crown[1], points_crown[5]], dtype=np.int32)
    cords_crown_triangle = np.array([[125, 94], [163, 98], [150, 135]], dtype=np.int32)
    cv2.fillConvexPoly(img, cords_crown_triangle1, 255)
    cv2.fillConvexPoly(img, cords_crown_triangle, 0)

    # Plot the square
    cords_square = np.array(
        [[50 - increase, 37.5 - increase], [100 + increase, 37.5 - increase], [100 + increase, 82.5 + increase],
         [50 - increase, 82.5 + increase]], dtype=np.int32)
    cords_square1 = np.array([[50, 37.5], [100, 37.5], [100, 82.5], [50, 82.5]], dtype=np.int32)
    cv2.fillConvexPoly(img, cords_square, 255)
    cv2.fillConvexPoly(img, cords_square1, 0)

    # Plot the circle
    cv2.circle(img, (190, 20), 15 + increase - 1, (255, 0, 0), -1)
    cv2.circle(img, (190, 20), 15 - 1, (0, 0, 0), -1)

    # Plot the ellipse
    cv2.ellipse(img, (140, 30), (15 + increase - 1, 6 + increase - 1), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (140, 30), (15 - 1, 6 - 1), 0, 0, 360, 0, -1)

    # Corner fix
    corner_cords1 = np.array([[0, 0], [increase, 0], [increase, 150], [0, 150]], dtype=np.int32)
    corner_cords2 = np.array([[0, 0], [250, 0], [250, increase], [0, increase]], dtype=np.int32)
    corner_cords3 = np.array([[250-increase, 0], [250, 0], [250, 150], [250-increase, 150]], dtype=np.int32)
    corner_cords4 = np.array([[0, 150-increase], [250, 150-increase], [250, 150], [0, 150]], dtype=np.int32)
    cv2.fillConvexPoly(img, corner_cords1, 255)
    cv2.fillConvexPoly(img, corner_cords2, 255)
    cv2.fillConvexPoly(img, corner_cords3, 255)
    cv2.fillConvexPoly(img, corner_cords4, 255)

    # resolution of the image
    # res = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return img


#######################################################################################################
# The section above this was just to plot and check workspace, the section below will be the algorithm implementation


class Node:
    def __init__(self, point):
        self.point = point
        self.cost = math.inf  # initially all the new nodes have infinite cost attached to them
        self.parent = None


class Robot:
    def __init__(self, dim, clear, int_pos, final_pos):
        self.dim = dim
        self.clear = clear
        self.int_pos = int_pos
        self.final_pos = final_pos


def cost_to_goal(point, goal_node_pos):
    point_x = point[0]
    point_y = point[1]
    goal_x = goal_node_pos[0]
    goal_y = goal_node_pos[1]
    euc_dist = np.sqrt((point_x-goal_x)**2 + (point_y-goal_y)**2)
    point = np.array(point)
    goal = np.array(goal_node_pos)
    euc_dist = np.linalg.norm(point - goal)
    return euc_dist


def pop_queue_element(queue):  # Priority Queue, outputs the node with least cost attached to it
    min_a = 0
    for elemt in range(len(queue)):
        if queue[elemt].cost < queue[min_a].cost:
            min_a = elemt
    return queue.pop(min_a)


def find_node(point, queue):
    for elem in queue:
        if elem.point == point:
            return queue.index(elem)
        else:
            return None


def move_up(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_y > 0 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x, point_y - 1]
        return new_point, base_cost
    else:
        return None, None


def move_down(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_y < 150 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x, point_y + 1]
        return new_point, base_cost
    else:
        return None, None


def move_left(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_x > 0 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x - 1, point_y]
        return new_point, base_cost
    else:
        return None, None


def move_right(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = 1
    if point_x < 250 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x + 1, point_y]
        return new_point, base_cost
    else:
        return None, None


def move_up_right(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x < 250 and point_y > 0 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x + 1, point_y - 1]
        return new_point, base_cost
    else:
        return None, None


def move_up_left(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x > 0 and point_y > 0 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x - 1, point_y - 1]
        return new_point, base_cost
    else:
        return None, None


def move_down_right(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x < 250 and point_y < 150 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x + 1, point_y + 1]
        return new_point, base_cost
    else:
        return None, None


def move_down_left(point, dimension, clearance, poly_points_updated):
    point_x = point[0]
    point_y = point[1]
    base_cost = np.sqrt(2)
    if point_x > 0 and point_y < 150 and not (call_obstacle_checks(point, dimension, clearance, poly_points_updated)):
        new_point = [point_x - 1, point_y + 1]
        return new_point, base_cost
    else:
        return None, None


def generate_node_location(action, current_point, dimension, clearance, poly_points_updated):
    if action == 'up':
        return move_up(current_point, dimension, clearance, poly_points_updated)
    if action == 'down':
        return move_down(current_point, dimension, clearance, poly_points_updated)
    if action == 'left':
        return move_left(current_point, dimension, clearance, poly_points_updated)
    if action == 'right':
        return move_right(current_point, dimension, clearance, poly_points_updated)
    if action == 'up_right':
        return move_up_right(current_point, dimension, clearance, poly_points_updated)
    if action == 'up_left':
        return move_up_left(current_point, dimension, clearance, poly_points_updated)
    if action == 'down_right':
        return move_down_right(current_point, dimension, clearance, poly_points_updated)
    if action == 'down_left':
        return move_down_left(current_point, dimension, clearance, poly_points_updated)


def count_entry_points(point):
    point_x = point[0]
    point_y = point[1]
    count = 0
    if point_y > 0:
        count += 1
    if point_y < 150:
        count += 1
    if point_x > 0:
        count += 1
    if point_x < 250:
        count += 1
    if point_x < 250 and point_y > 0:
        count += 1
    if point_x > 0 and point_y > 0:
        count += 1
    if point_x < 250 and point_y < 150:
        count += 1
    if point_x > 0 and point_y < 150:
        count += 1
    print("count is:", count)
    return count


def color_pixel(image_color, point):
    image_color[point[1], point[0]] = [0, 255, 255]
    return image_color


def track_back(node):
    print("Tracking Back")
    p = list()
    p.append(node.parent)
    parent = node.parent
    if parent is None:
        return p
    while parent is not None:
        p.append(parent)
        parent = parent.parent
    p_rev = list(p)
    return p_rev


def a_star_algo(image, robo1, resol):
    dimension = robo1.dim
    clearance = robo1.clear
    start_node_pos = robo1.int_pos
    goal_node_pos = robo1.final_pos

    image[start_node_pos[1], start_node_pos[0]] = [0, 255, 0]
    image[goal_node_pos[1], goal_node_pos[0]] = [0, 0, 255]
    start_node = Node(start_node_pos)
    start_node.cost = 0

    entry_points = count_entry_points(goal_node_pos)
    print("Entry points", entry_points)
    visited = list()
    queue = [start_node]
    actions = ["up", "down", "left", "right", "up_right", "down_right", "up_left", "down_left"]
    counter = 0

    while queue:
        current_node = pop_queue_element(queue)
        current_point = current_node.point
        visited.append(str(current_point))

        if counter == entry_points:
            return new_node.parent, image

        for action in actions:
            new_point, base_cost = generate_node_location(action, current_point, dimension, clearance,
                                                          poly_points_updated)
            if new_point is not None:
                if new_point == goal_node_pos:
                    if counter < entry_points:
                        counter += 1
                        print("Goal reached", counter)

                new_node = Node(new_point)
                new_node.parent = current_node

                image = color_pixel(image, current_node.point)
                image[start_node_pos[1], start_node_pos[0]] = [0, 255, 0]
                image[goal_node_pos[1], goal_node_pos[0]] = [0, 0, 255]

                resized_new_1 = cv2.resize(image, None, fx=resol, fy=resol, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Figure", resized_new_1)
                cv2.waitKey(1)

                if str(new_point) not in visited:
                    new_node.cost = base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos)
                    visited.append(str(new_node.point))
                    queue.append(new_node)
                else:
                    node_exist_index = find_node(new_point, queue)
                    if node_exist_index is not None:
                        temp_node = queue[node_exist_index]
                        if temp_node.cost > base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos):
                            temp_node.cost = base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos)
                            temp_node.parent = current_node
            else:
                continue
    return None, None


# Take user inputs for the robot size and initial position
print("This code will implement A* Algorithm for a Rigid Robot")
dimension = int(input("please enter the radius of the rigid robot: "))
clearance = int(input("please enter the desired clearance: "))
start_cords_x = int(input("please enter the starting x coordinate: "))
start_cords_y = int(input("please enter the starting y coordinate: "))
goal_cords_x = int(input("please enter the goal x coordinate: "))
goal_cords_y = int(input("please enter the goal y coordinate: "))


start_cords_y = 150 - start_cords_y
goal_cords_y = 150 - goal_cords_y

if start_cords_x > 250 or goal_cords_x > 250:
    print("The x coordinate should be within the limit of [0-250]")
    exit(0)

if start_cords_y < 0 or goal_cords_y < 0:
    print("The y coordinate should be within the limit of [0-150]")
    exit(0)

start_node_position = [start_cords_x, start_cords_y]
goal_node_position = [goal_cords_x, goal_cords_y]

# def __init__(self, dim, clear, int_pos, final_pos):
robo = Robot(dimension, clearance, start_node_position, goal_node_position)

poly_points_updated = new_poly_points(robo.dim, robo.clear)
image_int = plot_workspace(poly_points_updated, robo.dim, robo.clear)

if call_obstacle_checks(start_node_position, robo.dim, robo.clear, poly_points_updated):
    print("The start point of the robot can not be inside the obstacle")
    exit(0)

if call_obstacle_checks(goal_node_position, robo.dim, robo.clear, poly_points_updated):
    print("The goal point of the robot can not be inside the obstacle")
    exit(0)

result, image = a_star_algo(image_int, robo, 4)

if result is not None:
    nodes_list = track_back(result)
    for elem in nodes_list:
        x = elem.point[1]
        y = elem.point[0]
        image[x, y] = [0, 255, 0]

        resized_new = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Figure", resized_new)
        cv2.waitKey(100)
else:
    print("Sorry, result could not be reached")

print("Press any key to Quit")
cv2.waitKey(0)
cv2.destroyAllWindows()


