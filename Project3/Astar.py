import numpy as np
import sys
import cv2
import time
import math
import argparse

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
def algorithm(image,xi,yi,goal,distance, start_time, visual):

    visited=[]
    queue=[]
    visited_info=[]
    nodes_visited=[]
    cost_map=np.inf*np.ones((400,400))
    initial_pos = np.array([[xi, yi],[0,0]])
    goal_nodes = []
    pos_idx = 0

    queue.append(initial_pos)

    cost_map[initial_pos[0,0],initial_pos[0,1]] = 0

    goal_time = 1

    goal_position = [goal[0], goal[1]]

    while queue:

        min=0

        for i in range(len(queue)):
            if cost_map[queue[min][0][0],queue[min][0][1]]>cost_map[queue[i][0][0],queue[i][0][1]]:
                min=i
        
        current_node=queue.pop(min)
        current_position = [current_node[0,0],current_node[0,1]]
        
        parent_position = [current_node[1,0],current_node[1,1]]
        
        pos_idx = pos_idx + 1
        
        visited_info.append(current_node)
        
        nodes_visited.append(str(current_node[0]))

        image[200 - goal[1], goal[0]]=0

        for i in range (1,9):

            new_position, cost = check_move(i,current_position,distance)     

            if new_position is not False:

                image[200 - new_position[1], new_position[0]]=0
                
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

                new_cost=cost_map[current_position[0],current_position[1]]+cost

                # Counting the number of times the goal is reached (8)
                if new_position==goal_position:
                    print("Reached {} time".format(goal_time))
                    goal_time = goal_time + 1
                    goal_nodes.append([current_node, new_cost])

                if goal_time == 2:
                    return goal_nodes, visited_info

                
                if str(new_position) not in nodes_visited and cost_map[new_position[0],new_position[1]]>new_cost:
                    queue.append(np.array([new_position,current_position]))
                    cost_map[new_position[0],new_position[1]] = new_cost

                    for i in range(len(queue)):
                        if str(queue[i][0])==str(new_position) and str(queue[i][1])!=str(current_position):

                            queue[i]=np.array([new_position,current_position])
                            queue.pop(len(queue)-1)
            else:
                continue
    return goal_nodes, visited_info


def eucdist(current_pos, goal_pos):
    
    xi = current_pos[0]
    yi = current_pos[1]

    pos_cur = np.array([xi,yi])

    xg = goal_pos[0]
    yg = goal_pos[1]

    pos_goal = np.array([xg,yg])

    dist = np.linalg.norm(pos_goal - pos_cur)

    return dist

# Function to check the next move and output the cost of the movement along with the new position.
def check_move(act_number, current_pos, distance):

    temp_pos_x = current_pos[0]
    temp_pos_y = current_pos[1]

    if act_number==1:
        temp_pos_y = temp_pos_y - 1
        cost = 1 + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==2:
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1
        cost=math.sqrt(2) + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==3:
        temp_pos_x = temp_pos_x + 1
        cost=1 + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==4:
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2) + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==5:
        temp_pos_y = temp_pos_y + 1
        cost=1 + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==6:
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2) + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]

    if act_number==7:
        temp_pos_x = temp_pos_x - 1
        cost = 1 + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==8:
        # Top Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y - 1
        cost = math.sqrt(2) + eucdist(current_pos, goal_pos)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position,distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]   
    
    if temp_pos_x<0 or temp_pos_x>300 or temp_pos_y<0 or temp_pos_y>200:
        new_position=False

    return new_position,cost


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

    if (line1<=0 and line51>=0 and line52>=0) or (line2<=0 and line52<=0 and line53>=0) or (line3<=0 and line4>=0 and line53<=0) or (line5>=0 and line6<=0 and line51<=0):
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

    checkall = check0 or check1 or check2 or check3 or check4 or check5

    if checkall==True:
        return True

    else:
        return False


# Function for finding the goal with the least cost and backtracing the path from the goal to the input
def find_final_goal(goal_nodes, initial_pos, backinfo, image):

    print("Plotting path")
    goal_nodes = np.array(goal_nodes)
    min_idx = np.argmin(goal_nodes[:,1])
    
    final_node = goal_nodes[min_idx,0]

    print(goal_nodes)

    steps = 0

    parent = final_node[1]
    current = final_node[0]
    while not (parent[0]==0 and parent[1] == 0):
        for i in range(len(backinfo)):
            if str(parent)==str(backinfo[i][0]):
                new_node = backinfo[i]
                image[ 200 - new_node[1][1], new_node[1][0]]= 250
                steps = steps+1

        current = new_node[0]
        parent=new_node[1]

    resized_new_1 = cv2.resize(image, (640,480), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Figure", resized_new_1)
    cv2.waitKey(0)

    print("Number of steps taken are {}".format(steps))
    return None

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
    xi = 5
    yi = 5
    thetai = 0

    xf = goal[0]
    yf = goal[1]
    thetaf = goal[2]

    radius = 5
    clearance = 5

    distance = radius + clearance

    # xi = int(input("Please enter the input x coordinate of the rigid robot!"))
    # yi = int(input("Please enter the input y coordinate of the rigid robot!"))

    # xf = int(input("Please enter the input x coordinate of the rigid robot!"))
    # yf = int(input("Please enter the input y coordinate of the rigid robot!"))

    # radius = int(input("Please enter the input radius of the rigid robot!"))
    # clearance = int(input("Please enter the input y clearance of the rigid robot!"))

    # Accounting for distance from the boudaries, so the robot starts inside the workspace.
    xi = xi + distance
    yi = yi + distance

    xf = xf - distance
    yf = yf - distance

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

    # Passing the coordinates and the goal to the Djikstra algorithm
    goal_nodes, backinfo = algorithm(img, xi, yi, goal, distance, start_time, visual)

    # Finding and backtracing the path.
    find_final_goal(goal_nodes, initial_pos, backinfo, img)

    print("Total time taken is {} seconds ---".format(time.time() - start_time))

    return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--viz', help='For visualization set this to true')

    xf = 70
    yf = 80
    thetaf = 30

    goal_pos = [xf, yf, thetaf]

    main(goal_pos)