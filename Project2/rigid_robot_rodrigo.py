from __future__ import division
import numpy as np
import sys
import cv2
import time
import math

def space(distance):

    blank_image = 150*np.ones(shape=[200, 300, 1], dtype=np.uint8)

    cv2.circle(blank_image, (225, 50), 25 + distance, (0, 255, 0), 2)

    cv2.ellipse(blank_image, (150,100), (40 + distance, 20 + distance), 0, 0, 360, (0,255,0), 2) 

    rotrec = np.array([[[95,170], [95+5,170-9], [95+5-65,170-9 -38], [95-65,170-38]]], np.int32)

    weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)
    
    rhombus = np.array([[[225,190], [250, 175], [225, 160], [200, 175]]], np.int32)

    cv2.polylines(blank_image, rotrec, True, (0,255,0),2)

    cv2.polylines(blank_image, weird_shape, True, (0,255,0) ,2)

    cv2.polylines(blank_image, rhombus, True, (0,255,0), 2)

    rotrec1 = np.array([[[95 + 5 + (distance)/2, 170-9], [95+(distance/2), 170-13.5], [95, 170 + (distance)/2], [95 + (distance)/2, 170 + 2*np.sqrt(distance)]]], np.int32)

    rotrec2 = np.array([[[95 + (distance/2), 170-13.5], [95+5-65, 170-9 -38], [95+5-65 - (distance/2), 170-9-38 - (distance/2)], [95 + (distance/2), 170 - 13.5]]], np.int32)

    # rotrec3 = np.array([[[],[],[],[]]], np.int32)

    # rotrec4 = np.array([[[],[],[],[]]], np.int32)
    cv2.polylines(blank_image, rotrec1, True, (0,255,0), 2)
    cv2.polylines(blank_image, rotrec2, True, (0,255,0), 2)

    return blank_image


def algorithm(image,xi,yi, goal, radius, clearance):

    distance = radius + clearance
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

            new_position, cost = check_move(i,current_position, distance)     

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


def check_move(act_number, current_pos, distance):

    temp_pos_x = current_pos[0]
    temp_pos_y = current_pos[1]

    if act_number==1:
        # Upwar ACTION
        # temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==2:
        # Top Right ACTION
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==3:
        # RIght ACTION
        temp_pos_x = temp_pos_x + 1
        # temp_pos_y = temp_pos_y - 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==4:
        # Bottom Right ACTION
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==5:
        # Bottom ACTION
        # temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  

    if act_number==6:
        # Bottom Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y + 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  
    if act_number==7:
        # Left ACTION
        temp_pos_x = temp_pos_x - 1
        # temp_pos_y = temp_pos_y - 1
        cost=1
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]  


    if act_number==8:
        # Top Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y - 1
        cost=math.sqrt(2)
        new_position=[temp_pos_x,temp_pos_y]
        if check_movement(new_position, distance)==True:
            new_position=False
        else:
            new_position=[temp_pos_x,temp_pos_y]   
    
    if temp_pos_x<0 or temp_pos_x>300 or temp_pos_y<0 or temp_pos_y>200:
        new_position=False

    return new_position,cost


def check_rectangle(point, distance):
    
    x=point[0]
    y=point[1]
    
    coeff1=np.array(np.polyfit([95,100],[30,39],1))
    coeff2=np.array(np.polyfit([100,35],[39,77],1))
    coeff3=np.array(np.polyfit([35,30],[77,68],1))
    coeff4=np.array(np.polyfit([30,95],[68,30],1))
    line1 = round(y - coeff1[0] * x - coeff1[1]- distance*np.sin(np.arctan2(coeff1[0],1)))
    line2 = round(y - coeff2[0] * x - coeff2[1]+ distance*np.sin(np.arctan2(coeff2[0],1)))
    line3 = round(y - coeff3[0] * x - coeff3[1]+ distance*np.sin(np.arctan2(coeff3[0],1)))
    line4 = round(y - coeff4[0] * x - coeff4[1]- distance*np.sin(np.arctan2(coeff4[0],1)))

    if line1 >=0 and line2<=0 and line3<=0 and  line4>=0:
        return True
    else:
        return False

def check_rhombus(point, distance):
    x=point[0]
    y=point[1]

    coeff1=np.array(np.polyfit([225,250],[10,25],1))
    coeff2=np.array(np.polyfit([250,225],[25,40],1))
    coeff3=np.array(np.polyfit([225,200],[40,25],1))
    coeff4=np.array(np.polyfit([200,225],[25,10],1))
    line1 = round(y - coeff1[0] * x - coeff1[1]- distance*np.sin(np.arctan2(coeff1[0],1)))
    line2 = round(y - coeff2[0] * x - coeff2[1]+ distance*np.sin(np.arctan2(coeff2[0],1)))
    line3 = round(y - coeff3[0] * x - coeff3[1]+ distance*np.sin(np.arctan2(coeff3[0],1)))
    line4 = round(y - coeff4[0] * x - coeff4[1]- distance*np.sin(np.arctan2(coeff4[0],1)))

    if line1 >=0 and line2<=0 and line3<=0 and  line4>=0:
        return True
    else:
        return False


def check_circle(point, distance):
    x = point[0]
    y = point[1]
    dist = np.sqrt((x - 225)**2 + (y - 50)**2)
    if dist <= 25 + distance:
        return True
    else:
        return False


def check_ellipse(point, distance):
    a = 40 + distance
    b = 20 + distance
    x = point[0]
    y = point[1]
    dist = (((x-150)**2)/(a**2)) + (((y-100)**2)/(b**2))
    if dist <= 1:
        return True
    else:
        return False


# def check_poly(point, distance, max_x=300, max_y=200):

#     x = point[0]
#     y = point[1]
#     # weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)

#     coeff1 = np.polyfit([25, 75], [185, 185], 1)
#     coeff2 = np.polyfit([75, 100], [185, 150], 1)
#     coeff3 = np.polyfit([100, 75], [150, 120], 1)

#     coeff4 = np.polyfit([75, 50], [120, 150], 1)
    
#     coeff5 = np.polyfit([50, 20], [150, 120], 1)
    
#     coeff6 = np.polyfit([20, 25], [120, 185], 1)

#     coeff51 = np.polyfit([50,25],[150,185],1)
#     coeff52 = np.polyfit([50,75],[150,185],1)
#     coeff53 = np.polyfit([50,100],[150,150],1)

#     line1 = round(y - coeff1[0]*x - coeff1[1] + distance*np.sin(np.arctan2(coeff1[0],1)))
#     line2 = round(y - coeff2[0]*x - coeff2[1] + distance*np.sin(np.arctan2(coeff2[0],1)))
#     line3 = round(y - coeff3[0]*x - coeff3[1] + distance*np.sin(np.arctan2(coeff3[0],1)))
#     line4 = round(y - coeff4[0]*x - coeff4[1] - distance*np.sin(np.arctan2(coeff4[0],1)))
#     line5 = round(y - coeff5[0]*x - coeff5[1] - distance*np.sin(np.arctan2(coeff5[0],1)))
#     line6 = round(y - coeff6[0]*x - coeff6[1] - distance*np.sin(np.arctan2(coeff6[0],1)))

#     line51 = round(y - coeff51[0]*x - coeff51[1])
#     line52 = round(y - coeff52[0]*x - coeff52[1])
#     line53 = round(y - coeff53[0]*x - coeff53[1])

#     if line1<=0 and line2<=0 and line3>=0 and line6<=0 and (line5>=0 or line4>=0):
#             # check4 = True 
#             return True
#     return False

def check_poly(point, distance, max_x=300, max_y=200):

    x = point[0]
    y = point[1]
    # weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)

    coeff1 = np.polyfit([25, 75], [185, 185], 1)
    coeff2 = np.polyfit([75, 100], [185, 150], 1)
    coeff3 = np.polyfit([100, 75], [150, 120], 1)

    coeff4 = np.polyfit([75, 50], [120, 150], 1)

    coeff5 = np.polyfit([50, 20], [150, 120], 1)

    coeff6 = np.polyfit([20, 25], [120, 185], 1)

    coeff51 = np.polyfit([50,25],[150,185],1)
    coeff52 = np.polyfit([50,75],[150,185],1)
    coeff53 = np.polyfit([50,100],[150,150],1)

    coeff1[1]=coeff1[1]+distance*math.sin(np.pi/2-math.atan(coeff1[0]))
    coeff2[1]=coeff2[1]+distance*math.sin(np.pi/2-math.atan(coeff2[0]))
    coeff3[1]=coeff3[1]-distance*math.sin(np.pi/2-math.atan(coeff3[0]))
    coeff4[1]=coeff4[1]-distance*math.sin(np.pi/2-math.atan(coeff4[0]))
    coeff5[1]=coeff5[1]-distance*distance*math.sin(np.pi/2-math.atan(coeff5[0]))
    coeff6[1]=coeff6[1]+distance*math.sin(np.pi/2-math.atan(coeff6[0]))

    x1=(coeff2[1]-coeff1[1])/(coeff1[0]-coeff2[0])
    x2=(coeff3[1]-coeff2[1])/(coeff2[0]-coeff3[0])
    x3=(coeff4[1]-coeff3[1])/(coeff3[0]-coeff4[0])
    x4=(coeff5[1]-coeff4[1])/(coeff4[0]-coeff5[0])
    x5=(coeff6[1]-coeff5[1])/(coeff5[0]-coeff6[0])
    x6=(coeff1[1]-coeff6[1])/(coeff6[0]-coeff1[0])

    y1=x1*coeff1[0]+coeff1[1]
    y2=x2*coeff2[0]+coeff2[1]
    y3=x3*coeff3[0]+coeff3[1]
    y4=x4*coeff4[0]+coeff4[1]
    y5=x5*coeff5[0]+coeff5[1]
    y6=x6*coeff6[0]+coeff6[1]

    coeff1 = np.polyfit([x1, x2], [y1, y2], 1)
    coeff2 = np.polyfit([x2, x3], [y2, y3], 1)
    coeff3 = np.polyfit([x3, x4], [y3, y4], 1)

    coeff4 = np.polyfit([x4, x5], [y4,y5], 1)

    coeff5 = np.polyfit([x5, x6], [y5, y6], 1)

    coeff6 = np.polyfit([x6, x1], [y6, y1], 1)

    coeff51 = np.polyfit([x5,x1],[y5,y1],1)
    coeff52 = np.polyfit([x5,x2],[y5,y2],1)
    coeff53 = np.polyfit([x5,x3],[y5,y3],1)

    line1 = round(y - coeff1[0]*x - coeff1[1],1)
    line2 = round(y - coeff2[0]*x - coeff2[1],1)
    line3 = round(y - coeff3[0]*x - coeff3[1] ,1)
    line4 = round(y - coeff4[0]*x - coeff4[1],1)
    line5 = round(y - coeff5[0]*x - coeff5[1],1)
    line6 = round(y - coeff6[0]*x - coeff6[1])

    line51 = round(y - coeff51[0]*x - coeff51[1])
    line52 = round(y - coeff52[0]*x - coeff52[1])
    line53 = round(y - coeff53[0]*x - coeff53[1])

    if line1<=0 and line2<=0 and line3>=0 and line6<=0 and (line5>=0 or line4>=0):
            # check4 = True 
            return True
    return False


def check_if_not_in_image(position, max_x= 300, max_y=200):

    x = position[0]
    y = position[1]

    if x<=0 or x>=max_x or y<=0 or y>=max_y:
        return True

    else:
        return False


def check_movement(position, distance, max_x= 300, max_y=200):

    # print("Position inside checkvalid is {}".format(position))

    check0 = check_if_not_in_image(position, distance)

    check1 = check_poly(position, distance, max_x, max_y)
    check2 = check_rectangle(position, distance)
    check3 = check_circle(position, distance)
    check4 = check_ellipse(position, distance)    
    check5 = check_rhombus(position, distance)

    checkall = check0 or check1 or check2 or check3 or check4 or check5

    if checkall==True:
        print("There was an obstacle!")
        return True

    else:
        return False


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
                time.sleep(0.5)
                steps = steps+1

        current = new_node[0]
        parent=new_node[1]

    print("Number of steps taken are {}".format(steps))


def main():
    max_x = 300
    max_y = 200

    # Taking user Inputs
    xi = 5
    yi = 5
    xf = 295
    yf = 195

    radius = 5

    clearance = 5

    # xi = int(input("Please enter the input x coordinate of the point robot!"))
    # yi = int(input("Please enter the input y coordinate of the point robot!"))

    # xf = int(input("Please enter the input x coordinate of the point robot!"))
    # yf = int(input("Please enter the input y coordinate of the point robot!"))

    # radius = int(input("Please enter the radius of the robot!"))

    # clearance = int(input("Please enter the clearance of the robot!"))

    distance = radius + clearance

    img = space(distance)

    goal=[xf,yf]

    initial_pos = np.array([xi, yi])

    print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

    valid_init = check_movement(initial_pos, distance, max_x, max_y)
    valid_final = check_movement(goal, distance, max_x, max_y)

    if valid_init and valid_final != False:
        print("Please enter values that do not lie inside a shape!")
        return 0
    else:
        print(" Valid coordinates entered!")

    goal_nodes, backinfo = algorithm(img,xi,yi,goal, radius, clearance)

    find_final_goal(goal_nodes, initial_pos, backinfo, img, radius, clearance)


if __name__ == '__main__':
    main()






