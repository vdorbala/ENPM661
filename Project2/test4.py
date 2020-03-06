import numpy as np
import sys
import cv2
import pylsd
import time

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

def check_circle(point):
    x = point[0]
    y = point[1]
    dist = np.sqrt((x - 225)**2 + (y - 50)**2)
    if dist <= 25:
        return True
    else:
        return False

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

def check_poly(point, max_x, max_y):

    x = point[0]
    y = point[1]
    # weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)
    print(x,y)

    coeff1 = np.polyfit([25, 75], [185, 185], 1)
    coeff2 = np.polyfit([75, 100], [185, 150], 1)
    coeff3 = np.polyfit([100, 75], [150, 120], 1)

    coeff4 = np.polyfit([75, 50], [120, 150], 1)
    
    coeff5 = np.polyfit([50, 20], [150, 120], 1)
    
    coeff6 = np.polyfit([20, 25], [120, 185], 1)

    coeff51 = np.polyfit([50,25],[150,185],1)
    coeff52 = np.polyfit([50,75],[150,185],1)
    # coeff53 = np.polyfit([50,100],[150,150],1)

    line1 = round(y - coeff1[0]*x - coeff1[1])
    line2 = round(y - coeff2[0]*x - coeff2[1])
    line3 = round(y - coeff3[0]*x - coeff3[1])
    line4 = round(y - coeff4[0]*x - coeff4[1])
    line5 = round(y - coeff5[0]*x - coeff5[1])
    line6 = round(y - coeff6[0]*x - coeff6[1])

    line51 = round(y - coeff51[0]*x - coeff51[1])
    line52 = round(y - coeff52[0]*x - coeff52[1])
    # line53 = round(y - coeff53[0]*x - coeff53[1])


    print(line1, line2, line3, line4, line5, line6, line51, line52)

    if line1<=0 and line52>=0 and line51>=0:
        return True
    if line6>=0 and line5>=0 and line52<=0:
        return True
    if line52<=0 and line2<=0 and line3>=0 and line4>=0:
        return True

    return False

def space():

    blank_image = 150*np.ones(shape=[200, 300, 1], dtype=np.uint8)

    cv2.circle(blank_image, (225, 50), 25, (0, 255, 0), 2)

    cv2.ellipse(blank_image, (150,100), (40, 20), 0, 0, 360, (0,255,0), 2)

    rotrec = np.array([[[95,170],[95+5,170-9],[95+5-65,170-9 -38],[95-65,170-38]]], np.int32)
    
    weird_shape = np.array([[[25, 15], [75, 15], [100, 50], [75, 80], [50, 50], [20, 80]]], np.int32)
    
    rhombus = np.array([[[225,190], [250, 175], [225, 160], [200, 175]]], np.int32)

    cv2.polylines(blank_image, rotrec, True, (0,255,0),2)
    
    cv2.polylines(blank_image, weird_shape, True, (0,255,0) ,2)
    
    cv2.polylines(blank_image, rhombus, True, (0,255,0), 2)

    return blank_image


def check_move(act_number, current_pos, final_pos):

    temp_pos_x = current_pos[0]
    temp_pos_y = current_pos[1]

    if act_number==1:
        # Upwar ACTION
        # temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1


    if act_number==2:
        # Top Right ACTION
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y - 1

    if act_number==3:
        # RIght ACTION
        temp_pos_x = temp_pos_x + 1
        # temp_pos_y = temp_pos_y - 1

    if act_number==4:
        # Bottom Right ACTION
        temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1

    if act_number==5:
        # Bottom ACTION
        # temp_pos_x = temp_pos_x + 1
        temp_pos_y = temp_pos_y + 1

    if act_number==6:
        # Bottom Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y + 1

    if act_number==7:
        # Left ACTION
        temp_pos_x = temp_pos_x - 1
        # temp_pos_y = temp_pos_y - 1

    if act_number==8:
        # Top Left ACTION
        temp_pos_x = temp_pos_x - 1
        temp_pos_y = temp_pos_y - 1


    temp_pos = [temp_pos_x, temp_pos_y]
    # dist = np.linalg.norm(np.array(final_pos)-np.array(temp_pos))

    return dist, temp_pos

def check_actions(current_pos, final_pos):

    cost_min = 1000
    chosen_act = 0
    temp_fin = list(current_pos)

    for i in range(0,7):
        
        cost, temp_pos = check_move(i, current_pos, final_pos)

        if cost<cost_min:
            cost_min = cost
            temp_fin = list(temp_pos)

    return cost_min, temp_fin

def check_valid(position, max_x=300, max_y=200):

    check1 = check_poly(position, max_x, max_y)
    check2 = check_rectangle(position)
    check3 = check_circle(position)
    check4 = check_ellipse(position)    
    check5 = check_rhombus(position)

    checkall = check1 or check2 or check3 or check4 or check5

    if checkall==True:
        return False

    else:
        return True

def write_to_image(image, point):
    x = point[0]
    max_y = 200
    y = 200 - point[1]

    cv2.circle(image, (x, y), 1, (0, 255, 0), 2)


def check_nodes(c, arrays):

    verdict = any(np.array_equal(c, x) for x in arrays)

    return verdict
    # print("VERDICT IS {}".format(verdict))
    # if myarr in list_arrays:
    #     return True
    # else:
    #     return False

    # return next((True for elem in list_arrays if elem.size == myarr.size and np.allclose(elem, myarr)), False)


def assign_cost(current_pos, costmap, explored_pos):

    global i
    i = i+1
    x = current_pos[0]
    y = current_pos[1]

    if i == 1:
        current_cost = 0
    else:
        current_cost = costmap[x,y]

    newpos = (current_pos[0], current_pos[1]+1)
    a = [ list(item) for item in explored_pos]

    # print(current_cost)
    # and ((current_pos[0], current_pos[1]+1) in a == False)

    if (current_cost+1 < costmap[x,y+1]) and (check_nodes(np.array([current_pos[0],current_pos[1]+1]), explored_pos) == False):
        print("Entered 1")
        costmap[x,y+1] = current_cost + 1

    # print(current_cost+1)
    # print(explored_pos)
    if current_cost+np.sqrt(2) < costmap[x+1,y+1] and (check_nodes(np.array([current_pos[0]+1,current_pos[1]+1]), explored_pos) == False):
        print("Entered 2")
        costmap[x+1,y+1] = current_cost + np.sqrt(2)

    if current_cost+1 < costmap[x+1,y] and (check_nodes(np.array([current_pos[0]+1,current_pos[1]]), explored_pos) == False):
        print("Entered 3")
        costmap[x+1,y] = current_cost + 1

    if current_cost+np.sqrt(2) < costmap[x+1,y-1] and (check_nodes(np.array([current_pos[0]+1,current_pos[1]-1]), explored_pos) == False):
        print("Entered 4")
        costmap[x+1,y-1] = current_cost + np.sqrt(2)

    if current_cost+1 < costmap[x,y-1] and (check_nodes(np.array([current_pos[0],current_pos[1]-1]), explored_pos) == False):
        print("Entered 5")
        costmap[x,y-1] = current_cost + 1

    if current_cost+np.sqrt(2) < costmap[x-1,y-1] and (check_nodes(np.array([current_pos[0]-1,current_pos[1]-1]), explored_pos) == False):
        print("Entered 6")
        costmap[x-1,y-1] = current_cost + np.sqrt(2)

    if current_cost+1 < costmap[x-1,y] and (check_nodes(np.array([current_pos[0]-1,current_pos[1]]), explored_pos) == False):
        print("Entered 7")
        costmap[x-1,y] = current_cost + 1

    if current_cost+np.sqrt(2) < costmap[x-1,y+1] and (check_nodes(np.array([current_pos[0]-1,current_pos[1]+1]), explored_pos) == False):
        print("Entered 8")
        costmap[x-1,y+1] = current_cost + np.sqrt(2)

    # if check_nodes(np.array([current_pos[0],current_pos[1]]), explored_pos) == False:
    explored_pos.append(np.array(current_pos))

    return costmap, explored_pos

def main():

    img = space()
    
    max_x = 300
    max_y = 200

    [xi, yi] = [50, 50]
    [xf, yf] = [130, 150]

    # [xi, yi] = input("Please enter the starting coordinates of your robot as [x , y]! \n")

    # [xf, yf] = input("Please enter the final coordinates of your robot as [x , y]!\n")

    costmap = np.inf*np.ones((max_x, max_y))

    initial_pos = [xi, yi]
    final_pos = [xf, yf]

    print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

    valid = check_valid(initial_pos, max_x, max_y)

    if valid != True:
        print("Please enter values that do not lie inside a shape!")
        return 0
    else:
        print(" Valid coordinates entered!")


    costmap[initial_pos[0],initial_pos[1]] = 0

    current_pos = initial_pos

    explored_pos = []

    while current_pos[0] != final_pos[0] and current_pos[1] != final_pos[1]:

        costmap, explored_pos = assign_cost(current_pos, costmap, explored_pos)

        costmap[current_pos[0],current_pos[1]] = np.inf

        minimum = np.where(costmap == np.min(costmap))
        
        # print(minimum)

        # time.sleep(2)

        new_pos = zip(*minimum)[0]

        # print("Suggested new position is {}".format(new_pos))
        # print("explored_pos is {}".format(explored_pos))

        # time.sleep(2)

        if check_nodes(np.array([new_pos[0],new_pos[1]]), explored_pos) == False:
            if check_valid(new_pos)==True:
                current_pos = np.array(new_pos)

        # costmap[current_pos[0],current_pos[1]] = np.inf

        # print(np.min(costmap))

        img[current_pos[0],current_pos[1]]=0

        resized_new_1 = cv2.resize(img, (640,480), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        cv2.imshow("Figure", resized_new_1)
        
        cv2.waitKey(1)

        # write_to_image(img, current_pos)

    # cost = 1000

    # while(cost) > 5:
        # cost, temp_pos = check_actions(current_pos, final_pos)
        # current_pos = list(temp_pos)
        # write_to_image(img, current_pos)
        # print(cost)

    cv2.imshow('Final', img)
    cv2.waitKey(100)


if __name__ == '__main__':
    i = 0
    main()