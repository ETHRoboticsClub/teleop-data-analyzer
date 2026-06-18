tele-op data analyzer


Fetch teleop video from folder

provide a GUI with which one can see the main view camera in the middle and the wrist camera on the left and right side. 
With the key s one should be able to swap left and right hand wrist camera view (as some of them are accidently swapped).
With d one should be able to put the whole teleop sample(video, meta, data) in a folder discard
with g one should be able to put the whole teleop sample (video, meta, data) in a folder "good"
below the outputed camera view there should be plots for 
Joint velocity variance
Jerk (dddpos/dt³)
Gripper force profile
Action entropy per timestep
And there should be view of the robot doing the teleop action in mujoco or any other simulator.
so to clarify screen should be divided into 6 parts, top row the camera feed
low row the plots and the simulator view.
this file should be called teleop_data_selector_gui.py and should be placed in the teleop_data_analyzer folder.

Then there should be a script which plots Joint velocity variance
Jerk (dddpos/dt³)
Gripper force profile
Action entropy per timestep of all teleop data the same time. 
This file should be called teleop_data_analyzer_plotting.py 