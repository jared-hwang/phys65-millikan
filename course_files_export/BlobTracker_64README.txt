This is a readme.
Read the whole thing before you start.

***NOTE: This program has only been shown to work on Windows machines, it has
not been tested on Mac and there may be some compatibility issues with Linux***

Steps:

0) Install Anaconda (python 3) by going to: 
https://www.anaconda.com/distribution and selecting the Python 3.x version.
0.5) Install the OpenCV python package by using the Terminal/Command Prompt to
run the line: pip install opencv-python (you may have to replace 'pip' with 
'pip3')
0.75) Install the PyKalman python package by using the Terminal/Command Prompt 
to run the line:  conda install -c conda-forge pykalman 
1) Make sure you have the following files:
	i) The BlobTracker_64.py Python script
	ii) The savgol_error.py Python script

1.25) You will need a video file from the Millikan experiment, along with a csv
file containing data taken during the experiment on the state of the field. 
The current version of the script needs columns:
-------------------------------------------------------------------------------
sign of the voltage (1,0,-1) | magnitude of voltage (V) | time of split (sec)
e.g:        -1               |           468            |         50.723
-------------------------------------------------------------------------------
(see the given example file for a template)
This file MUST have the same name as the video file up to the extension.
(e.g. The file 'Millikan_example.mp4' will have the corresponding csv file
'Millikan_example.csv')

***Important Note: In order to link voltage changes to the videos, the onscreen
timer MUST be started before the start of the video, that is, the first frame 
of the video must show a time. If it does not, you will need to trim the video
so the timer is running when the video starts.***

***Another note: We haven't been able to get this to work with .mov files, so
you might need to convert them to .mp4 format before running the script.***

1.5) Place all scripts and videos and csv files into the same folder.
2) Open BlobTracker_64 in IDLE and run it, OR run BlobTracker_64 from
Terminal/Command Prompt.
2.5) If there is no calibration file present, the program will need to be
calibrated to your videos, follow the instructions to select a region of 
interest, etc. This only needs to be done once if all your videos look about
the same.
3) You will be asked to select a video in the folder.
4) You will be shown the first frame of the video, enter the time displayed on
the clock (in seconds). This is to calibrate the video timing with the program
timing.
5) Watch the video run. You do not have to watch particularly carefully (the 
tracked video will be saved for future review), but there are some things to 
watch out for that will save you time.
	Things to watch for:
		i) Take note of the IDs of droplets that seem really good
		ii) Take note of the IDs of droplets that cross each other
 		and steal each other's tracks (happens more often than you
 		might like) ESPECIALLY when it happens around a voltage
		switch
You might see a series of RuntimeWarning messages, ignore them.

6) When the video ends you will be shown a series of plots of the y-position 
and velocity of droplets that live for long enough (50 frames) and which live 
through at least one voltage change. It will be your job to select the ones 
that:
	A) Display a change in velocity when the voltage changes (to within a 
	certain tolerance) NOTE: The program handles both velocity changes with 
	change in direction and without change in direction (so look for 
	tracks that have either peaks or kinks in them)
	B) Have a visible flat(ish) velocity profile both before and after a 
	voltage change
	C) Aren't suspicious, where a suspicious droplet is one that:
		i) Changes velocity far from a voltage change
		ii) Lives for a suspiciously long time (several voltage 
		changes)
	   C2) If you think a droplet looks suspicious, open the _tracked.mp4
	   video to see if that droplet does anything weird.
	   C3) NOTE: Even if a droplet does something suspicious, it MIGHT 
	   still be usable as long as it doesn't do it near a voltage change
	   (See part 7)
7) After you pick out the good droplets, you will go through just those ones a 
second time to tell the program where to bracket its calculation of the mean
velocities. The program will take into account the times and voltages of the
electric field changes and display these on the plot: red vertical lines
indicate negative voltage, blue vertical lines indicate positive voltage, and
black vertical lines indicate 0 voltage.
	i) Look at each plot and determine the bounds of where the velocity is 
	relatively flat both before and after a SINGLE voltage change (at
	different levels). This does not need to be precise, but be sure to
	avoid any regions where you noticed the droplet do anything suspicious
	(you can look at the tracked video again for reference)
	ii) Close the plot and enter the times you determined into the 
	shell when it asks
		a) If you are unable to find flat regions in the velocity, or
		you notice that the droplet did something suspicious, you may
		veto the droplet by entering -1 for both the start and end
		times
	iii) After each droplet you will see a plot of the mean velocities it 
	found, for visual inspection, which you should close when ready.
8) The program will output a csv file containing the average velocities,
the errors on the velocities, and the strength of the electric field for each
droplet. The final calculation of charge will be left up to you.



--Ray Parker

If you have any questions, you can email me at rparkerparkerr@gmail.com