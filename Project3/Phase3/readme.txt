This folder contains all our scripts and results for Project 3 Phase 3.

The 'AstarPhase3.py' file has the implementation of A Star on a Turtlebot consideing the action states for the non-holonomic constraints.

In the 'main()' function of this file, defaults can be overwritten with user inputs by approproately changing the input variables (radius, clearance, xi,yf, thetai, xf, yf, thetaf,RPM1,RPM2). By default, these are set to the requested values.

Run the files by using:

"python AstarPhase3.py".

If a visualization is needed at each step, use,

"python AstarPhase3.py --viz True".

Otherwise, the plot containing the traceback path is viewed at the end of the script by default.

There will also be a vector plot shown at the end of the program. 
The vector output plot is shown in "vectorplot.png".

The time taken to run the algorithm will be shown after closing the vector plot.

With visuzalization, the script takes about 40 seconds to run.
Without visualization, it takes only around 15 seconds.

There are 3 output videos. They all have as inputs the same initial and goal nodes, but with variations in RPM velocities.

The videos are named as "RPM[RPM1,RPM2].mkv" from which RPM1 and RPM2 represent the velocities for each test.

Link to Github repo:
https://github.com/vdorbala/ENPM661/tree/master/Project3/Phase3
