This folder contains all our scripts and results.

The 'Astar_exp.py' file has the implementation of A Star on a rigid robot.

In the 'main()' function of this file, defaults can be overwritten with user inputs by approproately changing the input variables (radius, clearance, xi,yf, thetai, xf, yf, thetaf). By default, these are set to the requested values.

Run the files by using:

"python Astar_exp.py".

If a visualization is needed at each step, use,

"python Astar_exp.py --viz True".

Otherwise, the plot containing the traceback path is viewed at the end of the script by default.

There will also be a vector plot shown at the end of the program. This is however scaled up twice.
So the initial point of (30,50) will become (60,100), and the goal point of (150,150) will become (300,300).
The vector output plot is shown in "vectorplot.png".

The time taken to run the algorithm will be shown after closing the vector plot.

With visuzalization, the script takes about 1.5 hours to run.
Without visualization, it takes only around 25 minutes.

The "simulation_output.mp4" video contains the simulation video of A Star.

Link to Github repo:
https://github.com/vdorbala/ENPM661/tree/master/Project3