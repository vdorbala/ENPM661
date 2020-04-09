from __future__ import division
import numpy as np
import sys
import cv2
import time
import math
import argparse
import matplotlib.pyplot as plt


def space(distance):

    blank_image = 150*np.ones(shape=[1000, 1000, 1], dtype=np.uint8)

    # Drawing the boundary

    boundrec1 =  np.array([[[0, 0],[0, distance],[1000, distance],[1000,0]]], np.int32)
    boundrec2 =  np.array([[[0, 1000],[0, 1000 - distance],[1000, 1000 - distance],[1000,1000]]], np.int32)
    boundrec3 =  np.array([[[0, 0],[distance, 0],[distance, 1000],[0,1000]]], np.int32)
    boundrec4 =  np.array([[[1000 - distance, 0],[1000, 0],[1000, 1000],[1000 - distance, 1000 - distance]]], np.int32)

    cv2.polylines(blank_image, boundrec1, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec1, 255)

    cv2.polylines(blank_image, boundrec2, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec2, 255)

    cv2.polylines(blank_image, boundrec3, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec3, 255)

    cv2.polylines(blank_image, boundrec4, True, (0,255,0), 1)

    cv2.fillPoly(blank_image, boundrec4, 255)

    # Drawing the outer shapes

    cv2.circle(blank_image, (300, 800), 100+distance, (255, 255, 255), -1)
    cv2.circle(blank_image, (500, 500), 100+distance, (255, 255, 255), -1)
    cv2.circle(blank_image, (700, 800), 1000+distance, (255, 255, 255), -1)
    cv2.circle(blank_image, (700, 200), 1000+distance, (255, 255, 255), -1)

    cv2.rectangle(blank_image,(25-distance,575+distance),(175+distance,425-distance),(255,255,255), -1)
    cv2.rectangle(blank_image,(225-distance,275+distance),(375+distance,125-distance),(255,255,255), -1)
    cv2.rectangle(blank_image,(825-distance,575+distance),(975+distance,425-distance),(255,255,255), -1)




    cv2.rectangle(blank_image,(25,575),(175,425),(0,0,0), -1)
    cv2.rectangle(blank_image,(225,275),(375,125),(0,0,0), -1)
    cv2.rectangle(blank_image,(825,575),(975,425),(0,0,0), -1)
    
    cv2.circle(blank_image, (300, 800), 100, (0, 0, 0), -1)
    cv2.circle(blank_image, (500, 500), 100, (0, 0, 0), -1)
    cv2.circle(blank_image, (700, 800), 100, (0, 0, 0), -1)
    cv2.circle(blank_image, (700, 200), 100, (0, 0, 0), -1)

    return blank_image


def differential_drive(RPM,position):
	
	L=35
	r=35/2
	t=0
	dt=0.1
	x=position[0]
	y=position[1]
	theta=3.14*position[2]/180

	ul=RPM[0]
	ur=RPM[1]

	while t<1:
		t=t+dt
		x+=0.5*r*(ul+ur)*math.cos(theta)*dt
		y+=0.5*r*(ul+ur)*math.sin(theta)*dt
		theta+=(r/L)*(ul-ur)*dt

	theta=(180*theta/3.14)%360

	step_size=np.sqrt((x-position[0])**2+(y-position[1])**2)

	new_position=np.array([x,y,theta])

	

	return new_position,step_size



# Function to check if the points lie inside or outside the rectangle
def check_squares(point,distance):
	y=int(point[0])
	x=int(point[1])

	square1=(y>=425-distance)*(y<=575+distance)*(x>=25-distance)*(x<=175+distance)
	square2=(y>=425-distance)*(y<=575+distance)*(x>=825-distance)*(x<=975+distance)
	square3=(y>=725-distance)*(y<=875+distance)*(x>=225-distance)*(x<=375+distance)

	if square1 or square2 or square3 == 1:	
		return True
	else:
		return False


# Function to check if the points lie inside or outside the circle.
def check_circles(point,distance):

    x = int(point[0])
    y = int(point[1])

    dist = np.sqrt((x - 300)**2 + (y - 200)**2)
    circle1=(dist <= 100+distance)
    dist = np.sqrt((x - 500)**2 + (y - 500)**2)
    circle2=(dist <= 100+distance)
    dist = np.sqrt((x - 700)**2 + (y - 200)**2)
    circle3=(dist <= 100+distance)
    dist = np.sqrt((x - 700)**2 + (y - 800)**2)
    circle4=(dist <= 100+distance)
    if circle1 or circle2 or circle3 or circle4==1:
    	return True
    else:
    	return False


# Function to check if the points lie inside the frame of the image.
def check_if_not_in_image(position, distance, max_x= 1000, max_y=1000):

    x = int(position[0])
    y = int(position[1])

    if x<=distance or x>=max_x + distance or y<=distance or y>=max_y + distance ==True:
        return True

    else:
        return False


# Function for checking if the movements are valid or not
def check_movement(position, distance):

    check0 = check_if_not_in_image(position,distance)
    check1 = check_circles(position,distance)
    check2 = check_squares(position,distance)

    # print(check0, check1, check2,check3, check4, check5)

    checkall = check0 or check1 or check2

    if checkall==True:
        return True

    else:
        return False

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
def check_move(action, current_pos, distance):

    # Getting back the actual position from the index
    current_pos[0]= current_pos[0]*euc
    current_pos[1]= current_pos[1]*euc
    



    # New state containing the position and the angle of the checked action.
    new_position,step_size= differential_drive(action,current_pos)

    # Updating the cost with a uniform value of step size.
    cost = step_size

    # Checking if this movement lies inside an obstacle.
    if check_movement(new_position,distance)==True:
        new_position=False
    else:
    # Else, defining the new position according to the costmap visited node format.
        new_position = [int(new_position[0]/euc), int(new_position[1]/euc), int(new_position[2]/30)]


    # Returning the new position and the cost.
    return new_position, cost




# The Implementation of the A Star Algorithm
def algorithm(image,initial_pos, goal,distance, start_time, visual,actions):

    # Initial position and angle (50, 30, 60)
    xi = initial_pos[0]
    yi = initial_pos[1]
    thetai = initial_pos[2]



    # Visited node list
    queue=[]
    visited_info=[]
    nodes_visited=[]

    # Initializing costmap with infinite values
    cost_map=np.inf*np.ones((1100,1100,12),dtype=np.uint8)
    total_map=np.inf*np.ones((1100,1100,12),dtype=np.uint8)

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
    image[1000 - goal[1], goal[0]] = 0




    while queue:

        min=0

        # Check for minimum value of cost in queue, and assign that index to min.
        for i in range(len(queue)):
            if total_map[queue[min][0][0],queue[min][0][1], queue[min][0][2]]>total_map[queue[i][0][0],queue[i][0][1], queue[i][0][2]]:
                min=i

        # Getting current node
        current_node=queue.pop(min)

        print(current_node)

        # Getting current position and parent position from this node.
        current_position = [current_node[0,0], current_node[0,1], current_node[0,2]]
        
        # Appending the visited node list with the current node.
        visited_info.append(current_node)
        
        # Converted to string for easy comparison.
        nodes_visited.append(str(current_node[0]))


        for action in actions:

            new_position, cost = check_move(action, current_position, distance)


            if str(new_position) in nodes_visited:
                continue

            # Checking if new position is valid.
            if new_position is not False:

            	
                # Plotting the new position on the image.
                image[1000 - int(new_position[1]/2), int(new_position[0]/2)]=0
                
                resized_new_1 = cv2.resize(image, (640,640), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

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
                new_cost=cost_map[int(current_position[0]), int(current_position[1]), int(current_position[2])]+int(cost)


                # Updating the total cost with the euclidian distance heuristic.
                total_cost = new_cost + eucdist([new_position[0], new_position[1], new_position[2]], goal_position)

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

def main():

	args = parser.parse_args()

	visual = args.viz

	start_time = time.time()

	# xi=input("Enter initial x coordinate:  ")
	# yi=input("Enter initial y coordinate:  ")
	# thetai=input("Enter initial theta:  ")

	xi=100
	yi=100
	thetai=30

	initial_pos=np.array([xi,yi,thetai])

	# xf=input("Enter goal x coordinate:  ")
	# yf=input("Enter goal y coordinate:  ")
	# thetaf=input("Enter goal theta:  ")

	xf=150
	yf=200
	thetaf=30



	goal_pos=np.array([xf,yf,thetaf])

	#Input wheel velocities in RPM
	# RPM1=input("Enter first wheel RPM velocity: ")
	# RPM2=input("Enter second wheel RPM velocity: ")

	RPM1=5
	RPM2=10

	if visual==None:
		print("No visualization selected! There will only be a plot at the end! If you want visuals, add --viz True")

	#Stablishing action states of RPM velocity
	actions=[[0,RPM1],[RPM1,0],[RPM1,RPM1],[0,RPM2],[RPM2,0],[RPM2,RPM2],[RPM1,RPM2],[RPM2,RPM1]]

	#Turtlebot radius from datasheet
	radius=35/2

	#Wheel distance from datasheet, assumed 354 mm as in datasheet
	L=35

	#Clearance assumed 10 cm while minimum in datasheet 15mm

	clearance=10

	distance=int(radius+clearance)



	print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

	print("Chosen radius and clearance are, {} and {}".format(radius, clearance))

	img = space(distance)

    # Checking if the point coordinates are valid

	valid = check_movement(initial_pos, distance)

	valid1 = check_movement(goal_pos, distance)

	if (valid or valid1) != False:
		print("Please enter values that do not lie inside a shape!")
		return 0
	else:
		print(" Valid coordinates entered!")

	# Passing the coordinates and the goal to the A Star algorithm
	goal_nodes, backinfo = algorithm(img, initial_pos, goal_pos, distance, start_time, visual,actions)

	# Finding and backtracing the path.
	end_time = find_final_goal(goal_nodes, initial_pos, backinfo, img)

	print("Total time taken is {} seconds ---".format(end_time - start_time))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--viz', help='For visualization set this to true')


    # Euclidian Distance threshold
    euc = 0.5


    main()
