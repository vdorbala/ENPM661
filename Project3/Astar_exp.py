from __future__ import division
import numpy as np
import sys
import cv2
import time
import math
import argparse
import matplotlib.pyplot as plt

# Defining the workspace based on distance
def space(distance):

    blank_image = 150*np.ones(shape=[202, 302, 1], dtype=np.uint8)

    # Drawing the boundary

    boundrec1 =  np.array([[[0, 0],[0, distance],[300, distance],[300,0]]], np.int32)
    boundrec2 =  np.array([[[0, 200],[0, 200 - distance],[300, 200 - distance],[300,200]]], np.int32)
    boundrec3 =  np.array([[[0, 0],[distance, 0],[distance, 200],[0,200]]], np.int32)
    boundrec4 =  np.array([[[300 - distance, 0],[300, 0],[300, 200],[300 - distance, 300 - distance]]], np.int32)

    cv2.polylines(blank_image, boundrec1, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec1, 255)

    cv2.polylines(blank_image, boundrec2, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec2, 255)

    cv2.polylines(blank_image, boundrec3, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec3, 255)

    cv2.polylines(blank_image, boundrec4, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec4, 255)

    # Drawing the outer shapes

    cv2.circle(blank_image, (225, 50), 25+distance, (255, 255, 255), -1)

    cv2.ellipse(blank_image, (150,100), (40+distance, 20+distance), 
           0, 0, 360, (255,255,255),  thickness=-1, lineType=8, shift=0) 

    x1,x2,x3,x4,y1,y2,y3,y4=new_rec_points(distance)
    rotrec = np.array([[[x1,200-y1],[x2,200-y2],[x3,200-y3],[x4,200-y4]]], np.int32)
    cv2.polylines(blank_image, rotrec, True, (0,255,0),2)

    x1,x2,x3,x4,y1,y2,y3,y4=new_rhom_points(distance)
    rhombus = np.array([[[x1,200-y1], [x2, 200-y2], [x3,200-y3], [x4, 200-y4]]], np.int32)
    cv2.polylines(blank_image, rhombus, True, (0,255,0), 2)

    x1,x2,x3,x4,x5,x6,y1,y2,y3,y4,y5,y6=new_weird_points(distance)

    weird_shape = np.array([[[x1 + distance, 200-y1],[x2 + 5, 200-y2],[x3, 200-y3],[x4, 200 - y4 + distance],[x5 - distance, 200-y5 + distance],[x6 - distance, 200-y6]]],np.int32)

    cv2.polylines(blank_image, weird_shape, True, (0,255,0), 1)

    cv2.fillConvexPoly(blank_image, rotrec, 255)

    cv2.fillPoly(blank_image, weird_shape, 255)

    cv2.fillConvexPoly(blank_image, rhombus, 255)

    cv2.circle(blank_image, (225, 50), 25, (0, 0, 0), -1)

    cv2.ellipse(blank_image, (150,100), (40, 20), 0, 0, 360, (0,0,0), -1)

    # Drawing the inner shapes

    rotrec = np.array([[[95,170], [95+5,170-9], [95+5-65,170-9 -38], [95-65,170-38]]], np.int32)

    weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)
    
    rhombus = np.array([[[225,190], [250, 175], [225, 160], [200, 175]]], np.int32)

    cv2.polylines(blank_image, rotrec, True, (0,255,0),2)

    cv2.polylines(blank_image, weird_shape, True, (0,0,0), 1)

    cv2.polylines(blank_image, rhombus, True, (0,255,0), 2)

    cv2.fillConvexPoly(blank_image, rotrec, 0)

    cv2.fillPoly(blank_image, weird_shape, 0)

    cv2.fillConvexPoly(blank_image, rhombus, 0)

    return blank_image

# Defining new coefficients for rectangle
def new_rec_coeff(distance):
    coeff1=np.array(np.polyfit([95,100],[30,39],1))
    coeff2=np.array(np.polyfit([100,35],[39,77],1))
    coeff3=np.array(np.polyfit([35,30],[77,68],1))
    coeff4=np.array(np.polyfit([30,95],[68,30],1))


    coeff1[1]=coeff1[1]-distance*math.sin(1.57-math.atan(coeff1[0]))
    coeff2[1]=coeff2[1]+distance*math.sin(1.57-math.atan(coeff2[0]))
    coeff3[1]=coeff3[1]+distance*math.sin(1.57-math.atan(coeff3[0]))
    coeff4[1]=coeff4[1]-distance*math.sin(1.57-math.atan(coeff4[0]))

    return coeff1,coeff2,coeff3,coeff4

# Defining new points for rectangle
def new_rec_points(distance):
    
    coeff1,coeff2,coeff3,coeff4=new_rec_coeff(distance)

    x1=(coeff2[1]-coeff1[1])/(coeff1[0]-coeff2[0])
    x2=(coeff3[1]-coeff2[1])/(coeff2[0]-coeff3[0])
    x3=(coeff4[1]-coeff3[1])/(coeff3[0]-coeff4[0])
    x4=(coeff1[1]-coeff4[1])/(coeff4[0]-coeff1[0])
    

    y1=x1*coeff1[0]+coeff1[1]
    y2=x2*coeff2[0]+coeff2[1]
    y3=x3*coeff3[0]+coeff3[1]
    y4=x4*coeff4[0]+coeff4[1]

    return x1,x2,x3,x4,y1,y2,y3,y4

# Defining new rhombus coefficients
def new_rhom_coeff(distance):
    coeff1=np.array(np.polyfit([225,250],[10,25],1))
    coeff2=np.array(np.polyfit([250,225],[25,40],1))
    coeff3=np.array(np.polyfit([225,200],[40,25],1))
    coeff4=np.array(np.polyfit([200,225],[25,10],1))



    coeff1[1]=coeff1[1]-distance*math.sin(1.57-math.atan(coeff1[0]))
    coeff2[1]=coeff2[1]+distance*math.sin(1.57-math.atan(coeff2[0]))
    coeff3[1]=coeff3[1]+distance*math.sin(1.57-math.atan(coeff3[0]))
    coeff4[1]=coeff4[1]-distance*math.sin(1.57-math.atan(coeff4[0]))

    return coeff1,coeff2,coeff3,coeff4

# Defining new points for rhombus
def new_rhom_points(distance):
    
    coeff1,coeff2,coeff3,coeff4=new_rhom_coeff(distance)

    x1=(coeff2[1]-coeff1[1])/(coeff1[0]-coeff2[0])
    x2=(coeff3[1]-coeff2[1])/(coeff2[0]-coeff3[0])
    x3=(coeff4[1]-coeff3[1])/(coeff3[0]-coeff4[0])
    x4=(coeff1[1]-coeff4[1])/(coeff4[0]-coeff1[0])
    

    y1=x1*coeff1[0]+coeff1[1]
    y2=x2*coeff2[0]+coeff2[1]
    y3=x3*coeff3[0]+coeff3[1]
    y4=x4*coeff4[0]+coeff4[1]

    return x1,x2,x3,x4,y1,y2,y3,y4

# Defining new coefficients for polygon
def new_weird_coeff(distance):
    coeff1 = np.polyfit([25, 75], [185, 185], 1)
    coeff2 = np.polyfit([75, 100], [185, 150], 1)
    coeff3 = np.polyfit([100, 75], [150, 120], 1)
    coeff4 = np.polyfit([75, 50], [120, 150], 1)    
    coeff5 = np.polyfit([50, 20], [150, 120], 1)    
    coeff6 = np.polyfit([20, 25], [120, 185], 1)


    coeff1[1]=coeff1[1]+distance
    coeff2[1]=coeff2[1]+distance*math.sin(1.57+math.atan(coeff2[0]))
    coeff3[1]=coeff3[1]-distance*math.sin(1.57-math.atan(coeff3[0]))
    coeff4[1]=coeff4[1]-distance*math.sin(1.57+math.atan(coeff4[0]))
    coeff5[1]=coeff5[1]-distance*math.sin(1.57-math.atan(coeff5[0]))
    coeff6[1]=coeff6[1]+distance*math.sin(1.57-math.atan(coeff6[0]))
    
    return coeff1,coeff2,coeff3,coeff4,coeff5,coeff6  

# Defining new points for polygon
def new_weird_points(distance):

    coeff1,coeff2,coeff3,coeff4,coeff5,coeff6 = new_weird_coeff(distance)

    x1=(coeff2[1]-coeff1[1])/(coeff1[0]-coeff2[0])
    x2=(coeff3[1]-coeff2[1])/(coeff2[0]-coeff3[0])
    x3=(coeff4[1]-coeff3[1])/(coeff3[0]-coeff4[0])
    x4=(coeff5[1]-coeff4[1])/(coeff4[0]-coeff5[0])
    x5=(coeff6[1]-coeff5[1])/(coeff5[0]-coeff6[0])
    x6=(coeff6[1]-coeff1[1])/(coeff1[0]-coeff6[0])
    

    y1=x1*coeff1[0]+coeff1[1]
    y2=x2*coeff2[0]+coeff2[1]
    y3=x3*coeff3[0]+coeff3[1]
    y4=x4*coeff4[0]+coeff4[1]
    y5=x5*coeff5[0]+coeff5[1]
    y6=x6*coeff6[0]+coeff6[1]

    return x1,x2,x3,x4,x5,x6,y1,y2,y3,y4,y5,y6



# The Implementation of the A Star Algorithm
def algorithm(image,initial_pos, goal,distance, start_time, visual):

    # Initial position and angle (50, 30, 60)
    xi = initial_pos[0]
    yi = initial_pos[1]
    thetai = initial_pos[2]

    # Visited node list
    queue=[]
    visited_info=[]
    nodes_visited=[]

    # Initializing costmap with infinite values
    cost_map=np.inf*np.ones((400,400,12))
    total_map=np.inf*np.ones((400,400,12))

    # Initial position reformatted.
    # The theta value lies between (0 and 4)
    initial_pos = np.array([[int(xi/euc), int(yi/euc), int((thetai)/30)],[0,0,0]])

    goal_nodes = []
    pos_idx = 0

    # Append queue with initial position.
    queue.append(initial_pos)

    # Initialize the costmap at this position as 0.
    cost_map[initial_pos[0,0], initial_pos[0,1], initial_pos[0,2]] = 0
    total_map[initial_pos[0,0], initial_pos[0,1], initial_pos[0,2]] = 0

    goal_time = 1

    # Goal position also reformatted.
    goal_position = [int(goal[0]/euc), int(goal[1]/euc), int((goal[2])/30)]

    print("Goal position is {}".format([goal[0], goal[1]]))

    # Drawing the goal on the image.
    image[200 - goal[1], goal[0]] = 0


    while queue:

        min=0

        # Check for minimum value of cost in queue, and assign that index to min.
        for i in range(len(queue)):
            if total_map[queue[min][0][0],queue[min][0][1], queue[min][0][2]]>total_map[queue[i][0][0],queue[i][0][1], queue[i][0][2]]:
                min=i

        # Getting current node
        current_node=queue.pop(min)

        # Getting current position and parent position from this node.
        current_position = [current_node[0,0], current_node[0,1], current_node[0,2]]
        
        # Appending the visited node list with the current node.
        visited_info.append(current_node)
        
        # Converted to string for easy comparison.
        nodes_visited.append(str(current_node[0]))


        for i in range (-60,61,30):

            new_position, cost = check_move(i, current_position, distance)     
            
            if str(new_position) in nodes_visited:
                continue

            # Checking if new position is valid.
            if new_position is not False:

                # Plotting the new position on the image.
                image[200 - int(new_position[1]/2), int(new_position[0]/2)]=0
                
                resized_new_1 = cv2.resize(image, (640,480), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

                # Condition used to make plotting faster
                if visual!=None:
                    if (pos_idx%50)==0:
                        # print("--- {} seconds ---".format(time.time() - start_time))
                        cv2.imshow("Figure", resized_new_1)
                        cv2.waitKey(10)
                else:
                    if pos_idx%100==0:
                        print("--- {} seconds ---".format(time.time() - start_time))

                # Updating the cost of the new position.
                new_cost=cost_map[current_position[0], current_position[1], current_position[2]]+cost

                # Updating the total cost with the euclidian distance heuristic.
                total_cost = new_cost + eucdist([new_position[0], new_position[1], new_position[2]], goal_pos)

                # Counting the number of times the goal is reached (12)
                if new_position[0]==goal_position[0] and new_position[1]==goal_position[1]:
                    print("Reached {} time".format(goal_time))
                    goal_time = goal_time + 1
                    goal_nodes.append([current_node, new_cost])

                if goal_time == 2:
                    return goal_nodes, visited_info

                # Checking if new position lies in the visited node list or if the new cost is lower than the current cost of the new position.
                if total_map[new_position[0],new_position[1], new_position[2]]>total_cost:
                    
                    # Appending the queue with the new position and the current (parent) position
                    queue.append(np.array([new_position,current_position]))
                    
                    # Updating the costmap cost to the cost of the new position.
                    cost_map[new_position[0], new_position[1], new_position[2]] = new_cost
                    total_map[new_position[0], new_position[1], new_position[2]] = total_cost

                    # Finding the new state in the queue, and replacing it with the current new position. Then popping the last element.
                    for i in range(len(queue)):
                        if str(queue[i][0])==str(new_position) and str(queue[i][1])!=str(current_position):

                            queue[i]=np.array([new_position,current_position])
                            queue.pop(len(queue)-1)
            else:
                continue
    return goal_nodes, visited_info

# Function to compute euclidian distance
def eucdist(current_pos, goal_pos):
    
    xi = current_pos[0]*euc
    yi = current_pos[1]*euc
    ang = current_pos[2]*(np.pi/6)

    pos_cur = np.array([xi,yi])

    xg = goal_pos[0]
    yg = goal_pos[1]

    pos_goal = np.array([xg,yg])

    dist = np.linalg.norm(pos_goal - pos_cur)

    # Computing Angle cost to add to the heuristic.
    dir_vec = ([yg - yi, xg - xi])/np.sqrt((yg-yi)**2 + (xg-xi)**2)

    cur_pos = [np.sin(ang), np.cos(ang)] 

    angle_cost = (-1*np.dot(dir_vec,cur_pos) + 1)/(2*dist)

    return dist

# Function to check the next move and output the cost of the movement along with the new position.
def check_move(act_number, current_pos, distance):

    # Getting back the actual position from the index
    temp_pos_x = current_pos[0]*euc
    temp_pos_y = current_pos[1]*euc
    temp_angle = current_pos[2]

    # Getting the possible position based on the step size and the action (theta)
    temp_pos_x = round((temp_pos_x + step_size*np.cos(temp_angle + act_number)))
    temp_pos_y = round((temp_pos_y + step_size*np.sin(temp_angle + act_number)))

    temp_angle = (temp_angle+ act_number)%360

    # Updating the cost with a uniform value of step size.
    cost = step_size

    # New state containing the position and the angle of the checked action.
    new_position = [temp_pos_x, temp_pos_y, temp_angle]

    # Checking if this movement lies inside an obstacle.
    if check_movement(new_position,distance)==True:
        new_position=False
    else:
    # Else, defining the new position according to the costmap visited node format.
        new_position = [int(temp_pos_x/euc), int(temp_pos_y/euc), int(temp_angle/30)]

    # Checking if the new position lies outside the box
    if temp_pos_x<0 or temp_pos_x>300 or temp_pos_y<0 or temp_pos_y>200:
        new_position=False

    # Returning the new position and the cost.
    return new_position, cost


# Function to check if the points lie inside or outside the rectangle
def check_rectangle(point,distance):

    x=point[0]
    y=point[1]

    coeff1,coeff2,coeff3,coeff4=new_rec_coeff(distance)

    line1 = round(y - coeff1[0] * x - coeff1[1])
    line2 = round(y - coeff2[0] * x - coeff2[1])
    line3 = round(y - coeff3[0] * x - coeff3[1])
    line4 = round(y - coeff4[0] * x - coeff4[1])

    if line1 >=0 and line2<=0 and line3<=0 and  line4>=0:
        return True
    else:
        return False


# Function to check if the points lie inside or outside the rhombus
def check_rhombus(point,distance):
    
    x=point[0]
    y=point[1]

    coeff1,coeff2,coeff3,coeff4=new_rhom_coeff(distance)

    line1 = round(y - coeff1[0] * x - coeff1[1])
    line2 = round(y - coeff2[0] * x - coeff2[1])
    line3 = round(y - coeff3[0] * x - coeff3[1])
    line4 = round(y - coeff4[0] * x - coeff4[1])

    if line1 >=0 and line2<=0 and line3<=0 and line4>=0:
        return True
    else:
        return False


# Function to check if the points lie inside or outside the circle.
def check_circle(point,distance):

    x = point[0]
    y = point[1]
    dist = np.sqrt((x - 225)**2 + (y - 50)**2)
    if dist <= 25+distance:
        return True
    else:
        return False


# Function to check if the points lie inside or outside the ellipse.
def check_ellipse(point,distance):

    a = 40+distance
    b = 20+distance
    x = point[0]
    y = point[1]
    dist = (((x - 150) ** 2) / (a ** 2)) + (((y - 100) ** 2) / (b ** 2))-1
    if dist <=0:
        return True
    else:
        return False


# Function to check if the points lie inside or outside the polygon. 
def check_poly(point, distance, max_x=300, max_y=200):

    x = point[0]
    y = point[1]
    
    coeff1,coeff2,coeff3,coeff4,coeff5,coeff6 = new_weird_coeff(distance)
    x1,x2,x3,x4,x5,x6,y1,y2,y3,y4,y5,y6=new_weird_points(distance)

    coeff51 = np.polyfit([x5,x1],[y5,y1],1)
    coeff52 = np.polyfit([x5,x2],[y5,y2],1)
    coeff53 = np.polyfit([x5,x3],[y5,y3],1)

    line1 = round(y - coeff1[0]*x - coeff1[1])
    line2 = round(y - coeff2[0]*x - coeff2[1])
    line3 = round(y - coeff3[0]*x - coeff3[1])
    line4 = round(y - coeff4[0]*x - coeff4[1])
    line5 = round(y - coeff5[0]*x - coeff5[1])
    line6 = round(y - coeff6[0]*x - coeff6[1])

    line51 = round(y - coeff51[0]*x - coeff51[1])
    line52 = round(y - coeff52[0]*x - coeff52[1])
    line53 = round(y - coeff53[0]*x - coeff53[1])

   # Checking the half planes if the point lies inside or outside. 

    if line1<=0 and line52>=0 and line51>=0:
       return True
    if line6>=0 and line5>=0 and line52<=0:
       return True
    if line52<=0 and line2<=0 and line3>=0 and line4>=0:
       return True

    return False

# Function to check if the points lie inside the frame of the image.
def check_if_not_in_image(position, distance, max_x= 300, max_y=200):

    x = position[0]
    y = position[1]

    if x<=0 + distance or x>=max_x + distance or y<=0 + distance or y>=max_y + distance:
        return True

    else:
        return False


# Function for checking if the movements are valid or not
def check_movement(position, distance, max_x= 300, max_y=200):

    check0 = check_if_not_in_image(position, distance)
    check1 = check_poly(position, distance, max_x, max_y)
    check2 = check_rectangle(position,distance)
    check3 = check_circle(position,distance)
    check4 = check_ellipse(position,distance)    
    check5 = check_rhombus(position,distance)

    # print(check0, check1, check2,check3, check4, check5)

    checkall = check0 or check1 or check2 or check3 or check4 or check5

    if checkall==True:
        return True

    else:
        return False

# Function for finding the goal with the least cost and backtracing the path from the goal to the input
def find_final_goal(goal_nodes, initial_pos, backinfo, image):

    print("Plotting path")
    goal_nodes = np.array(goal_nodes)
    print(goal_nodes)
    min_idx = np.argmin(goal_nodes[:,1])
    
    final_node = goal_nodes[min_idx,0]

    fig, ax = plt.subplots()

    steps = 0

    parent = final_node[1]
    current = final_node[0]
    while not (parent[0]==0 and parent[1] == 0):
        for i in range(len(backinfo)):
            if str(parent)==str(backinfo[i][0]):
                new_node = backinfo[i]
                # Not plotting the initial node of the costmap (0,0)
                print(new_node[0][0],new_node[0][1], new_node[1][0], new_node[1][1])
                if new_node[1][0]!=0:
                    # Plotting Arrow
                    plt.arrow(new_node[0][0], new_node[0][1], new_node[1][0] - new_node[0][0], new_node[1][1] - new_node[0][1],
                     head_width=0.5, head_length=0.3, fc='k', ec='k')
                steps = steps+1

        current = new_node[0]
        parent = new_node[1]

    end_time = time.time()
    plt.grid()

    ax.set_aspect('equal')

    plt.xlim(0,300)
    plt.ylim(0,300)

    plt.title('Vector Plot',fontsize=10)

    plt.savefig('vectorploy.png', bbox_inches='tight')

    plt.show()
    plt.close()

    print("Number of steps taken are {}".format(steps))
    return end_time

def main(goal):

    args = parser.parse_args()

    visual = args.viz

    if visual==None:
        print("No visualization selected! There will only be a plot at the end! If you want visuals, add --viz True")

    max_x = 300
    max_y = 200

    start_time = time.time()

    # Taking user Inputs

    # Defaults
    xi = 50
    yi = 30
    thetai = 60

    xf = goal[0]
    yf = goal[1]
    thetaf = goal[2]

    radius = 1
    clearance = 1

    distance = radius + clearance

    initial_pos = np.array([xi, yi, thetai])
    
    print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

    print("Chosen radius and clearance are, {} and {}".format(radius, clearance))

    img = space(distance)

    # Checking if the point coordinates are valid

    valid = check_movement(initial_pos, distance, max_x, max_y)

    valid1 = check_movement(goal, distance, max_x, max_y)

    if (valid and valid1) != False:
        print("Please enter values that do not lie inside a shape!")
        return 0
    else:
        print(" Valid coordinates entered!")

    # Passing the coordinates and the goal to the A Star algorithm
    goal_nodes, backinfo = algorithm(img, initial_pos, goal, distance, start_time, visual)

    # Finding and backtracing the path.
    end_time = find_final_goal(goal_nodes, initial_pos, backinfo, img)

    print("Total time taken is {} seconds ---".format(end_time - start_time))

    return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--viz', help='For visualization set this to true')

    # Goal coordinates and angle
    xf = 150
    yf = 150
    thetaf = 0

    goal_pos = [xf, yf, thetaf]

    # Euclidian Distance threshold
    euc = 0.5

    # Defining the step size
    step_size = 1

    main(goal_pos)