import numpy as np
import sys
import cv2
import pylsd

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


def check_valid(initial_pos,final_pos):
	pass

def space():
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

	dist = np.linalg.norm(np.array(final_pos)-np.array(temp_pos))

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


def check_valid(position, image):

	pass


def write_to_image(image, point):
	x = point[0]
	y = point[1]

	cv2.circle(image, (y, x), 1, (0, 255, 0), 2)

def main():

	img = space()
	# print(defset.shape)
	[xi, yi] = [78, 50]
	[xf, yf] = [130, 250]

	# [xi, yi] = input("Please enter the starting coordinates of your robot as [x , y]! \n")

	# [xf, yf] = input("Please enter the final coordinates of your robot as [x , y]!\n")

	initial_pos = [xi, yi]
	final_pos = [xf, yf]

	print("Chosen initial and final coordinates are, [{} {}] and [{} {}]".format(xi, yi, xf, yf))

	# check_valid(initial_pos, img)

	write_to_image(img, final_pos)

	current_pos = initial_pos

	cost = 1000

	while(cost) > 5:
		cost, temp_pos = check_actions(current_pos, final_pos)
		current_pos = list(temp_pos)
		write_to_image(img, current_pos)
		print(cost)

	cv2.imshow('Final', img)
	cv2.waitKey(0)


if __name__ == '__main__':
	main()
