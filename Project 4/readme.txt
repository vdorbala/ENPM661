The objective of Project 4 is to simulate a baxter robot moving two blocks from one table to another skipping obstacles in the path.
The content of this folder is:
  "Object_scenes_rviz": this folder contains he .scene models to compute the obstacles in rviz
  "ik_pick_and_place_demo.py": this is the code that publishes the way points to create the path in ros
  "baxter_demo.mkv": this video show the final result of the robot simulation in Gazebo

To view the video of the processes of obtaining the way points with rviz and rostopic go to the following link:

  https://drive.google.com/drive/folders/1JMASz8g4mGBpq2lKYTVl6NcfSTIfAnWq?usp=sharing

The video is speed up x8

To get the way points it was first estimated which position coordinates to command to grab the blocks. This was done by taken as reference the initial positions defined in the gazebo models of each block and then iterating to get a more precisse offset of the coordinates of that position. The following way points were identified using rviz and rostopic as it can be seen in the video starting from minute 2 
