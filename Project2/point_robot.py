import numpy as np
import sys
import cv2
import time
import math

# Defining the workspace

def space():

    increase = 10

    blank_image = 150*np.ones(shape=[200, 300, 1], dtype=np.uint8)

    cv2.circle(blank_image, (225, 50), 25, (0, 255, 0), 2)

    cv2.ellipse(blank_image, (150,100), (40, 20), 
           0, 0, 360, (0,255,0), 2) 

    rotrec = np.array([[[95,170],[95+5,170-9],[95+5-65,170-9 -38],[95-65,170-38]]], np.int32)
    weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)
    rhombus = np.array([[[225,190], [250, 175], [225, 160], [200, 175]]], np.int32)

    cv2.polylines(blank_image, rotrec, True, (0,255,0),2)
    cv2.polylines(blank_image, weird_shape, True, (0,255,0) ,2)
    cv2.polylines(blank_image, rhombus, True, (0,255,0), 2)

    return blank_image

# The Implementation of the Djikstra Algorithm

def algorithm(image,xi,yi, goal):

    visited=[]
    queue=[]
    visited_info=[]
    nodes_visited=[]
    cost_map=np.inf*np.ones((200,300))
    initial_pos = np.array([[xi, yi],[0,0]])
    goal_nodes = []
    pos_idx = 0

    queue.append(initial_pos)

    
    cost_map[initial_pos[0,0],initial_pos[0,1]]=0


    time = 1

    while queue:

        min=0

        for i in range(len(queue)):
            if cost_map[queue[min][0][0],queue[min][0][1]]>cost_map[queue[i][0][0],queue[i][0][1]]:
                min=i
        
        current_node=queue.pop(min)
        current_position=[current_node[0,0],current_node[0,1]]
        
        parent_position=[current_node[1,0],current_node[1,1]]
        
        pos_idx = pos_idx + 1
        
        visited_info.append(current_node)
        
        nodes_visited.append(str(current_node[0]))

        image[200 - goal[1], goal[0]]=0

        for i in range (1,9):

            new_position, cost = check_move(i,current_position)     

            if new_position is not False:

                image[200 - new_position[1], new_position[0]]=0
                resized_new_1 = cv2.resize(image, (640,480), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Figure", resized_new_1)
                cv2.waitKey(1)

                new_cost=cost_map[current_position[0],current_position[1]]+cost
                # print(new_position)
                if new_position==goal:
                    print("Reached {} time".format(time))
                    time = time + 1
                    goal_nodes.append([current_node, new_cost])

                if time == 9:
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
    return None


# Function to check the next move and output the cost of the movement along with the new position.
def check_move(act_number, current_pos):

    temp_pos_x = current_pos[0]
    temp_pos_y = current_pos[1]

    if act_number==1:
        temp_pos_y = temp_pos_y - 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==2:
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==3:
        temp_pos_x = temp_pos_x + 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==4:
        # Bottom Right ACTION
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==5:
        # Bottom ACTION
        # temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==6:
        # Bottom Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  
    if act_number==7:
        # Left ACTION
        temp_pos_x = temp_pos_x - 1
        # temp_pos_y = temp_pos_y - 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==8:
        # Top Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y - 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]   
    
    if temp_pos_x<0 or temp_pos_x>300 or temp_pos_y<0 or temp_pos_y>200:
        new_position=False

    return new_position,cost

# Function to check if the points lie inside or outside the rectangle
def check_rectangle(point):
    x=point[0]
    y=point[1]
    coeff1=np.array(np.polyfit([95,100],[30,39],1))
    coeff2=np.array(np.polyfit([100,35],[39,77],1))
    coeff3=np.array(np.polyfit([35,30],[77,68],1))
    coeff4=np.array(np.polyfit([30,95],[68,30],1))
    line1 = round(y - coeff1[0] * x - coeff1[1])
    line2 = round(y - coeff2[0] * x - coeff2[1])
    line3 = round(y - coeff3[0] * x - coeff3[1])
    line4 = round(y - coeff4[0] * x - coeff4[1])

    if line1 >=0 and line2<=0 and line3<=0 and  line4>=0:
        return True
    else:
        return False

# Function to check if the points lie inside or outside the rhombus
def check_rhombus(point):
    x=point[0]
    y=point[1]

    coeff1=np.array(np.polyfit([225,250],[10,25],1))
    coeff2=np.array(np.polyfit([250,225],[25,40],1))
    coeff3=np.array(np.polyfit([225,200],[40,25],1))
    coeff4=np.array(np.polyfit([200,225],[25,10],1))
    line1 = round(y - coeff1[0] * x - coeff1[1])
    line2 = round(y - coeff2[0] * x - coeff2[1])
    line3 = round(y - coeff3[0] * x - coeff3[1])
    line4 = round(y - coeff4[0] * x - coeff4[1])

    if line1 >=0 and line2<=0 and line3<=0 and  line4>=0:
        return True
    else:
        return False

# Function to check if the points lie inside or outside the circle.
def check_circle(point):
    x = point[0]
    y = point[1]
    dist = np.sqrt((x - 225)**2 + (y - 50)**2)
    if dist <= 25:
        return True
    else:
        return False

# Function to check if the points lie inside or outside the ellipse.
def check_ellipse(point):
    a = 40
    b = 20
    x = point[0]
    y = point[1]
    dist = (((x-150)**2)/(a**2)) + (((y-100)**2)/(b**2))
    if dist <= 1:
        return True
    else:
        return False

# Function to check if the points lie inside or outside the polygon. 
def check_poly(point, max_x=300, max_y=200):

    x = point[0]
    y = point[1]

    coeff1 = np.polyfit([25, 75], [185, 185], 1)
    coeff2 = np.polyfit([75, 100], [185, 150], 1)
    coeff3 = np.polyfit([100, 75], [150, 120], 1)

    coeff4 = np.polyfit([75, 50], [120, 150], 1)
    
    coeff5 = np.polyfit([50, 20], [150, 120], 1)
    
    coeff6 = np.polyfit([20, 25], [120, 185], 1)

    coeff51 = np.polyfit([50,25],[150,185],1)
    coeff52 = np.polyfit([50,75],[150,185],1)
    coeff53 = np.polyfit([50,100],[150,150],1)

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

    if line1<=0 and line2<=0 and line3>=0 and line6<=0 and (line5>=0 or line4>=0):
            return True
    return False

# Function to check if the points lie inside the frame of the image.
def check_if_not_in_image(position, max_x= 300, max_y=200):

    x = position[0]
    y = position[1]

    if x<=0 or x>=max_x or y<=0 or y>=max_y:
        return True

    else:
        return False

# Function for checking if the movements are valid or not
def check_movement(position, max_x= 300, max_y=200):

    check0 = check_if_not_in_image(position)

    check1 = check_poly(position, max_x, max_y)
    check2 = check_rectangle(position)
    check3 = check_circle(position)
    check4 = check_ellipse(position)    
    check5 = check_rhombus(position)

    checkall = check0 or check1 or check2 or check3 or check4 or check5

    if checkall==True:
        print("There was an obstacle!")
        return True

    else:
        return False

# Function for finding the goal with the least cost and backtracing the path from the goal to the input
def find_final_goal(goal_nodes, initial_pos, backinfo, image):

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
                resized_new_1 = cv2.resize(image, (640,480), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                cv2.imshow("Figure", resized_new_1)
                cv2.waitKey(100)
                time.sleep(3)
                steps = steps+1

        current = new_node[0]
        parent=new_node[1]

    print("Number of steps taken are {}".format(steps))



def main():

    img = space()

    max_x = 300
    max_y = 200

    start_time = time.time()

    # Taking User Input
    xi = int(input("Please enter the input x coordinate of the point robot!"))
    yi = int(input("Please enter the input y coordinate of the point robot!"))

    xf = int(input("Please enter the input x coordinate of the point robot!"))
    yf = int(input("Please enter the input y coordinate of the point robot!"))

    goal=[xf,yf]

    initial_pos = np.array([xi, yi])
    
    print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

    # Checking if the point coordinates are valid

    valid = check_movement(initial_pos, max_x, max_y)

    if valid != False:
        print("Please enter values that do not lie inside a shape!")
        return 0
    else:
        print(" Valid coordinates entered!")

    # Passing the coordinates and the goal to the Djikstra algorithm
    goal_nodes, backinfo = algorithm(img,xi,yi,goal)

    # Finding and backtracing the path.
    find_final_goal(goal_nodes, initial_pos, backinfo, img)

    print("--- {} seconds ---".format(time.time() - start_time))

if __name__ == '__main__':
    main()






