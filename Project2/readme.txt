The Code folder contains all our scripts.

The 'point_robot.py' file has the implementation of Dijakstra on a point robot.
The 'rigid_robot.py' file has the implementation of Dijakstra on a rigid robot.

In both these files, in the 'main()' function, defaults can be overwritten with user inputs by approproately uncommenting the input lines.

Run the files by using:

"python point_robot.py" for the point robot, and
"python rigid_robot.py" for the rigid robot.

The plot containing the traceback path is viewed at the end of the script. This was done to save time. Press any key after the plot shows up to close it.

The 'rigid_robot.py' script takes about 9 minutes to run, while the 'point_robot.py' script takes 15 minutes,
for the given goal points from (5,5) to (295, 195).