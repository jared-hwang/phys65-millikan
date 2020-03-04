#Blob tracker for the Millikan oil drop experiment
#by Ray Parker (2019-2020)

print('loading packages, please wait')
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
import cv2
import tkinter as tk #for file-picker and future GUI implementation
from tkinter.filedialog import askopenfilename
import os
import sys
from scipy.signal import savgol_filter
from savgol_error import savgol_error
from matplotlib.widgets import Button
from matplotlib.transforms import blended_transform_factory
import datetime

from pykalman import KalmanFilter
print('done loading packages')
print('')

root = tk.Tk()
root.withdraw()



xweight = 1
yweight = 1

#these are parameters you can tweak if you're having a hard time tracking droplets
greediness = 10
speediness = 10
velocityweight = 0.5

overlapThresh = 0

unlinked_count_thresh = 5
duplicate_count_thresh = 3



#tweak these parameters to make sure droplets show up in the mask
threshmin = 180
threshmax = 255




droplet_container = None





if len(sys.argv)>1:
    filein = sys.argv[1]
else:
    filein = askopenfilename()
if not os.path.isfile(filein):
    print("It looks like you haven't specified the correct file path, please double check it.")
    raise RuntimeError("Input file not found! Check path!")


def findCenter(cont):
    M = cv2.moments(cont)
    cX = M["m10"]/M["m00"]
    cY = M["m01"]/M["m00"]

    return (cX, cY)



#shamelessly stolen from: https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
def overlapArea(boxa, boxb):  # returns -1 if rectangles don't intersect
    dx = min(boxa[0]+boxa[2], boxb[0]+boxb[2]) - max(boxa[0], boxb[0])
    dy = min(boxa[1]+boxa[3], boxb[1]+boxb[3]) - max(boxa[1], boxb[1])
    if (dx>=0) and (dy>=0):
        return abs(dx*dy)
    else:
        return -1

#stringifies a time given in seconds
def sec2str(t):
    return str(int(t//3600)).zfill(2)+":"+str(int((t-t//3600)//60)).zfill(2)+":"+str(int(np.round(t-((t-t//3600)//60)))).zfill(2)+"."+str(int(np.modf(t)[0]*100)).zfill(2)


def unpack_droplet(droplet):
    states=np.array(droplet.filtered_states)
    xpos, xvel, ypos, yvel = [states[:,i] for i in range(states.shape[1])]
    cov = np.array(droplet.filtered_covariances)
    birth_time = droplet.birth_time
    lifetime = len(xpos)*dt
    return xpos, ypos, xvel, yvel, cov, birth_time, lifetime

def cumulative_average(a):
    return np.add.accumulate(a)/np.arange(1,len(a)+1)



class ROI:
    def __init__(self, im, winname='Click and Drag to Select ROI'):
        self.im = im
        self.winname = winname
        self.x = 0
        self.y = 0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.w = 0
        self.h = 0
        self.draw = False
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, im)
        cv2.setMouseCallback(winname, self.selectROI)
        k = cv2.waitKey(0)
        if k==27:
            self.x = 0
            self.y = 0
            self.w = im.shape[1]
            self.h = im.shape[0]
        cv2.destroyWindow(winname)


    def selectROI(self, event, mx, my, flags, param):
        imcopy = np.copy(self.im)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1, self.y1 = mx, my
            self.draw = True
        if event == cv2.EVENT_MOUSEMOVE:
            self.x2, self.y2 = mx, my
            if self.draw:
                cv2.rectangle(imcopy, (self.x1,self.y1),(self.x2,self.y2), (255,0,0), thickness=2)
            else:
                cv2.rectangle(imcopy, (self.x,self.y),(self.x+self.w,self.y+self.h), (255,0,0), thickness=2)
                
        if event == cv2.EVENT_LBUTTONUP:
            self.x = min(self.x1, self.x2)
            self.y = min(self.y1, self.y2)
            self.w = abs(self.x1-self.x2)
            self.h = abs(self.y1-self.y2)
            self.draw = False
        cv2.imshow(self.winname, imcopy)
        
    def getCoords(self):
        return (self.x,self.y,self.w,self.h)

#Ruler class for drawing and measuring distances
class Ruler:

    def __init__(self, im, winname="Click and Drag a Known Distance"):
        self.im = im
        self.winname = winname
        self.d = 0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.draw = False
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, im)
        cv2.setMouseCallback(winname, self.measureDist)
        cv2.waitKey(0)
        cv2.destroyWindow(winname)
        
    def measureDist(self, event, mx, my, flags, param):
        imcopy = np.copy(self.im)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1, self.y1 = mx, my
            self.draw = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                self.x2, self.y2 = mx, my
                cv2.line(imcopy, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 255), thickness=2)
                cv2.imshow(self.winname, imcopy)

        if event == cv2.EVENT_LBUTTONUP:
            self.draw = False
            self.d = np.sqrt(np.power(np.abs(self.x2-self.x1),2)+np.power(np.abs(self.y2-self.y1),2))

    def getDist(self):
        return self.d




def calibrate_video(frame):
    print('')
    print("Please click and drag on the image to select only the area where droplets will fall. \
Do NOT include any part of the window frame or overlays on the video. Press ENTER when done.`")
    roi = ROI(frame, "Click and Drag to Select Region of Interest")
    (x, y, w, h) = roi.getCoords()

    dpxs = []
    Ngrids = []

    print('\n')
    print("Please click and drag VERTICALLY on the image to draw the ruler over a known number of grid lines (count them!). \
You must do this five times, preferably over long scales in different places.")
    print("It is very important that you only draw along the Y direction because the video may be squashed in one direction. \
Only the Y velocity will count in the end, so please only calibrate vertically.")
    for i in range(5):
        ruler = Ruler(frame, f"Click and Drag Over the Grid ({i+1}/5)")
        dpx = ruler.getDist()
        Ngrid = int(input('Please enter the number of squares you just drew over. (Integer values only!): '))
        print('')
        dpxs.append(dpx)
        Ngrids.append(Ngrid)

    dpxs = np.array(dpxs)
    Ngrids = np.array(Ngrids)

    dgrid = float(input("Please enter the distance between gridlines in MILLIMETERS: "))

    return x, x+w, y, y+h, dpxs, Ngrids, dgrid


class Droplet:

    
    def __init__(self, pos, framenum, t, vel=[0,0], box=None, dropletid=0, contourid=None):


        self.birth_frame = framenum
        self.birth_time = t
        
        self.obs_pos = np.array(pos)
        _, _, self.w, self.h = box
        self.dropletid = dropletid

        self.contourid = contourid

        self.unlinked = False
        self.unlinked_count = 0

        self.duplicate = False
        self.duplicate_count = 0

        self.filt_pos = np.array(pos)
        self.filt_vel = np.array([0,-self.h/self.w])
        self.filt_dims = np.array([self.w, self.h])
        
        self.observation_matrix = observation_matrix
        self.observation_offset = np.zeros(2)


        self.transition_covariance = np.diag([1,1,speediness,1])*greediness


        self.kf = KalmanFilter(transition_matrices=transition_matrix2d,
                               transition_offsets=transition_offset,
                               transition_covariance=self.transition_covariance,
                               observation_matrices=self.observation_matrix,
                               observation_offsets=self.observation_offset,
                               observation_covariance=self.make_observation_covariance_matrix(),
                               initial_state_mean=self.make_state_vector(),
                               initial_state_covariance=greediness*np.identity(4))


        self.filtered_states = [self.make_state_vector()]
        self.filtered_covariances = [self.transition_covariance]



##    def setPosition(self,pos):
##        self.pos = pos
##    def setVelocity(self,vel):
##        self.vel = vel
    def setDimensions(self,w,h):
        self.w = w
        self.h = h


    def update_observation(self, pos, box, contourid=None):
        #print(f'updating droplet {self.dropletid} with contour {contourid}')
        self.last_obs = self.obs_pos
        if pos is None:
            self.obs_pos = np.ma.masked_array([0,0],mask=True)
        else:
            self.obs_pos = pos

            
        if box is not None:
            self.setDimensions(*box[2:])

        self.contourid = contourid

        if contourid is None:
            self.unlinked = True
            self.unlinked_count += 1
        else:
            self.unlinked = False
            self.unlinked_count = 0

            
        if not np.ma.is_masked(self.last_obs) and not np.ma.is_masked(self.obs_pos):
            #print("doing the whole thingy")
            self.obs_vel = (self.obs_pos-self.last_obs)/dt
            self.observation_matrix = observation_matrix_vel
            self.observation_offset = np.zeros(4)

            observation = np.array([pos[0], self.obs_vel[0], pos[1], self.obs_vel[1]])

        else:
            #print("doing half the thingy")
            self.obs_vel = None
            self.observation_matrix = observation_matrix
            observation = self.obs_pos
            self.observation_offset = np.zeros(2)





        self.update_kf(observation)
        
        
        


    def make_observation_covariance_matrix(self):
        if self.observation_matrix.shape==(2,4):
            return 2*np.array([[self.w**2, 0],[0, self.h**2]])
        else:
            return np.array([[self.w**2,                  0,         0,                  0],
                             [        0, velocityweight*self.obs_vel[0]**2,         0,                  0],
                             [        0,                  0, self.h**2,                  0],
                             [        0,                  0,         0, velocityweight*self.obs_vel[1]**2]])*2

    def make_state_vector(self):
        return np.array([self.filt_pos[0], self.filt_vel[0], self.filt_pos[1], self.filt_vel[1]])


    def update_kf(self, observation):
        #print(observation)
        filtered_state_i, filtered_cov_i = self.kf.filter_update(self.filtered_states[-1],
                                                            self.filtered_covariances[-1],
                                                            observation=observation,
                                                            transition_matrix=self.get_transition_matrix(),
                                                            transition_covariance=self.transition_covariance,
                                                            transition_offset=self.get_transition_offset(),
                                                            observation_covariance=self.make_observation_covariance_matrix(),
                                                            observation_matrix=self.observation_matrix,
                                                            observation_offset=self.observation_offset)
        self.filtered_states.append(filtered_state_i)
        self.filtered_covariances.append(filtered_cov_i)

        self.filt_pos = filtered_state_i[[0,2]]
        self.filt_vel = filtered_state_i[[1,3]]

        self.filt_dims = 2*np.sqrt(np.diag(filtered_cov_i))[[0,2]]
                                                            
                                                            

    def get_transition_matrix(self):
        return transition_matrix2d

    def get_transition_offset(self):
        return transition_offset



def pick_droplets(saved_droplets):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax_button_save = fig.add_axes([0.1,0.01,0.2, 0.04])
    ax_button_toss = fig.add_axes([0.4,0.01,0.2, 0.04])

    button_save = Button(ax_button_save, 'Save Droplet', color='lightcyan', hovercolor='0.975')
    button_toss = Button(ax_button_toss, 'Toss Droplet', color='red', hovercolor='0.975')

    interesting_droplets = []


    pos_plot, = ax1.plot([], [], label='tracked position')
    pos_plot_smooth, = ax1.plot([], [], label='smoothed track')
    ax1.legend()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('position [pixels]')

    vel_plot, = ax2.plot([],[], label='computed velocity')
    vel_plot_smooth, = ax2.plot([],[])

    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('velocity [pixels/sec]')


    transform = blended_transform_factory(ax1.transData, ax1.transAxes)

    ax1.vlines(split_times, 0, 1, colors=line_colors, linestyles='dashed', transform=transform)

    transform = blended_transform_factory(ax2.transData, ax2.transAxes)

    ax2.vlines(split_times, 0, 1, colors=line_colors, linestyles='dashed', transform=transform)



    def plot_next():
        global droplet_container
        if len(saved_droplets)>0:
            droplet = saved_droplets.pop(0)
            droplet_container = droplet
            xpos, ypos, xvel, yvel, cov, birth_time, lifetime = unpack_droplet(droplet)
            ypos = ymax-ypos
            yvel = -yvel
            ysmooth = savgol_filter(ypos, 31, 2)

            yerr = np.sqrt(cov[:,2,2])
            yvel_err = np.sqrt(cov[:,3,3])

            xpos = xpos-xmin

            yvel_ = savgol_filter(ypos, 31, 4, deriv=1, delta=dt)#np.diff(ysmooth)/dt
##            yvel_smooth = savgol_filter(yvel, 21, 1)
            
            time_array = np.linspace(birth_time, birth_time+lifetime, len(ypos))
            if np.any((split_times>birth_time)&(split_times<birth_time+lifetime)):
                pos_plot.set_data(time_array, ypos)

                pos_plot_smooth.set_data(time_array, ysmooth)

                vel_plot.set_data(time_array, yvel_)
##                vel_plot_smooth.set_data(time_array, yvel_smooth)

                ax1.set_xlim([birth_time, birth_time+lifetime])
                ax2.set_xlim([birth_time, birth_time+lifetime])

                ax1.set_ylim([np.min(ypos), np.max(ypos)])
                ax2.set_ylim([np.min(yvel_), np.max(yvel_)])

                fig.suptitle(f"droplet id: {droplet.dropletid}, time: {datetime.timedelta(seconds=droplet.birth_time)}")

                fig.canvas.draw_idle()

                return droplet
            return plot_next()
        else:
            plt.close()
            print('')
            print("all done")
            return None



    button_save.on_clicked(lambda e: [print('saving droplet',droplet_container.dropletid),interesting_droplets.append(droplet_container),plot_next()])

    button_toss.on_clicked(lambda e: plot_next())
    plt.tight_layout()
    plt.subplots_adjust(top=0.94,bottom=0.165)
    plot_next()
    plt.show()

    return interesting_droplets


def analyze_droplets(droplets):
    voffs = []
    vons = []
    voff_errs = []
    von_errs = []
    Vs = []
    for i in range(len(droplets)):

        droplet = droplets[i]

        fig, axes = plt.subplots(1,2,sharex=True)

        xpos, ypos, xvel, yvel, cov, birth_time, lifetime = unpack_droplet(droplet)
        ypos = (ymax-ypos)*px2mm
        ysmooth = savgol_filter(ypos, 31, 2)

        yvel_smooth = savgol_filter(ypos, 21, 1)
        yerr = np.sqrt(cov[:,2,2])*px2mm
##        yvel_err = np.sqrt(cov[:,3,3])

##        yvel_err = np.sqrt(cov[:,3,3])*px2mm#[np.sqrt(yerr[i]**2+yerr[i+1]**2)/dt for i in range(len(yerr)-1)]
##        yvel_err.append(yvel_err[-1])
##
##        yvel_err = np.array(yvel_err)

        yvel_ = savgol_filter(ypos, 31, 4, deriv=1, delta=dt)#np.diff(ysmooth)/dt
        yacc = savgol_filter(ypos, 31, 2, deriv=2, delta=dt)#np.diff(ysmooth)/dt

        yvel_err = savgol_error(ypos, yerr, 31, 4, deriv=1, delta=dt)

        
        time_array = np.linspace(birth_time, birth_time+lifetime, len(ypos))

        axes[0].plot(time_array, yvel_)
        axes[0].fill_between(time_array, yvel_-yvel_err, yvel_+yvel_err, alpha=0.7)
        axes[1].plot(time_array, yacc)

        transform = blended_transform_factory(axes[0].transData, axes[0].transAxes)

        axes[0].vlines(split_times, 0, 1, colors=line_colors, linestyles='dashed', transform=transform)

        transform = blended_transform_factory(axes[1].transData, axes[1].transAxes)

        axes[1].vlines(split_times, 0, 1, colors=line_colors, linestyles='dashed', transform=transform)

        plt.xlim(np.min(time_array),np.max(time_array))
        axes[0].set_ylim(np.min(yvel_), np.max(yvel_))
        axes[1].set_ylim(np.min(yacc), np.max(yacc))
        print('')
        print("Please pick out an approximate start and end time around ONE voltage change, then close the plot. Be prepared to enter these numbers.")
        print("You may veto any droplet by entering a negative start or end time.")

        axes[0].set_xlabel('time [s]')
        axes[0].set_ylabel('velocity [mm/s]')
        axes[1].set_ylabel('acceleration [mm/s^2]')
        fig.suptitle(f"droplet id: {droplet.dropletid}")
        plt.tight_layout()
        plt.show()

        tmin = float(input("time start: "))
        tmax = float(input("time end: "))
        print('')

        if tmin<0 or tmax<0:
            print(f'droplet {droplet.dropletid} vetoed!')
            continue

        switch_index = np.argwhere((tmin<split_times)&(tmax>split_times))[0][0]

        time_switch = split_times[switch_index]
        line_color = line_colors[switch_index]
        switch_voltage = split_voltages[switch_index] if split_voltages[switch_index]!=0 else split_voltages[switch_index-1]


##        v_left_median = np.median(yvel_[(tmin<=time_array)&(time_switch>=time_array)])
##        v_right_median = np.median(yvel_[(tmax>=time_array)&(time_switch<=time_array)])

        left = (tmin<=time_array)&(time_switch>=time_array)
        right = (tmax>=time_array)&(time_switch<=time_array)
        
        yacc_left = yacc[left]
        left_weights = np.exp(-np.abs(yacc_left)/np.median(np.abs(yacc_left)))
        left_weights = left_weights/np.sum(left_weights)
        yacc_right = yacc[right]
        right_weights = np.exp(-np.abs(yacc_right)/np.median(np.abs(yacc_right)))
        right_weights = right_weights/np.sum(right_weights)


        v_left_mean = np.sum(left_weights*yvel_[left]/yvel_err[left]**2)/np.sum(left_weights/yvel_err[left]**2)
        v_right_mean = np.sum(right_weights*yvel_[right]/yvel_err[right]**2)/np.sum(right_weights/yvel_err[right]**2)
##        v_left_mean = np.average(yvel_[left], weights = left_weights)
##        v_right_mean = np.average(yvel_[right], weights = right_weights)

        v_left_err = np.sqrt(np.sum(left_weights**2/yvel_err[left]**2)/np.sum(left_weights/yvel_err[left]**2)**2)
        v_right_err = np.sqrt(np.sum(right_weights**2/yvel_err[right]**2)/np.sum(right_weights/yvel_err[right]**2)**2)

##        v_left_err = np.sqrt(1/np.sum(1/yvel_err[left]**2))
##        v_right_err = np.sqrt(1/np.sum(1/yvel_err[right]**2))


##        print(v_left_mean,'+/-',v_left_err)
##        print(v_right_mean,'+/-',v_right_err)

##        print(v_left_median, v_left_mean, np.mean(yvel_[(tmin<=time_array)&(time_switch>=time_array)]))
##        print(v_right_median, v_right_mean, np.mean(yvel_[(tmax>=time_array)&(time_switch<=time_array)]))
####
        plt.plot(time_array, yvel_)
        plt.xlabel('time [s]')
        plt.ylabel('velocity [mm/s]')
        plt.title(f"droplet id: {droplet.dropletid}")
        plt.hlines([v_left_mean], tmin, time_switch, colors='m',linestyles=['dashed'])
        plt.fill_between([tmin,time_switch],[v_left_mean-v_left_err]*2, [v_left_mean+v_left_err]*2, color='m', alpha=0.7)
        plt.hlines([v_right_mean], time_switch, tmax, colors='g',linestyles=['dashed'])
        plt.fill_between([time_switch,tmax],[v_right_mean-v_right_err]*2, [v_right_mean+v_right_err]*2, color='g', alpha=0.7)


        plt.show()

        if line_color!='k':
            voffs.append(v_left_mean)
            voff_errs.append(v_left_err)
            vons.append(v_right_mean)
            von_errs.append(v_right_err)
        else:
            voffs.append(v_right_mean)
            voff_errs.append(v_right_err)
            vons.append(v_left_mean)
            von_errs.append(v_left_err)

        Vs.append(switch_voltage) #V/m


    voffs = np.array(voffs)
    vons = np.array(vons)
    voff_errs = np.array(voff_errs)
    von_errs = np.array(von_errs)
    Vs = np.array(Vs)
    return voffs/1000, vons/1000, voff_errs/1000, von_errs/1000, Vs #converting from mm/s to m/s








# Selects and reads video

vidcap = cv2.VideoCapture(filein)

datafile = filein[:-4]+'.csv'

field_data = np.genfromtxt(datafile, delimiter=',', comments=None, skip_header=1)
split_voltages = field_data[:,1]*field_data[:,0] #V
line_colors = ['b' if split_voltages[i]>0 else 'r' if split_voltages[i]<0 else 'k' for i in range(len(split_voltages))] #for plotting
split_times = field_data[:,2] #s




frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = 14.985
#  vidcap.get(cv2.CAP_PROP_FPS) returns int(29.97/2) = 14 for stupid stupid reasons.
#  See: https://www.youtube.com/watch?v=3GJUM6pCpew







dt = 3/framerate #Framerate of video is three times the framerate of the camera


m=1
k=1
g=1

#Matrix that takes a [x_i, vx_i] vector to [x_(i+1), vx_(i+1)]
transition_matrix = np.array([[1,        dt],
                              [0,  1]]) #-k/m*dt

#putting two of them together for x and y (4x4)
transition_matrix2d = np.concatenate((np.concatenate((transition_matrix, \
                                                      np.zeros_like(transition_matrix))), \
                                      np.concatenate((np.zeros_like(transition_matrix),\
                                                      transition_matrix))), axis=1)

#Droplets accelerate downwards at g
transition_offset = np.array([0, 0, 0, g*dt])  #x, vx, y, vy


#We only observe the position of each droplet (x, y)
observation_matrix = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0]])

#If we COULD observe the velocity this would be our matrix
observation_matrix_vel = np.identity(4)

 
    
frameCount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
if not frameCount:
    raise RuntimeError(f"Video file '{filein}' could not be read! Please try again.")




##success, prevframe = vidcap.read()
##prevframe = prevframe[ymin:ymax,xmin:xmax,:]
##prevframe = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
##prevframe = cv2.inRange(prevframe, threshmin,threshmax)
##prevframe = cv2.GaussianBlur(prevframe, (5,5), 0)

##vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameCount-1500)
success, frame = vidcap.read()

try:
    xmin, xmax, ymin, ymax, px2mm, stdpx2mm = np.load('calibration.npy', allow_pickle=False)
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
except FileNotFoundError:
    xmin, xmax, ymin, ymax, ypx, Ngrid, dgrid = calibrate_video(frame)
    px2mm = np.mean(Ngrid/ypx*dgrid) #measured 15 divisions at 0.1mm per division
    stdpx2mm = np.std(Ngrid/ypx*dgrid)
    np.save('calibration.npy', [xmin, xmax, ymin, ymax, px2mm, stdpx2mm], allow_pickle=False)






cv2.namedWindow('First Frame -- See Shell Window', cv2.WINDOW_NORMAL)
cv2.imshow('First Frame -- See Shell Window', frame)
cv2.waitKey(1)
t0 = float(input("Please enter the time on the clock (in seconds): "))
cv2.destroyWindow('First Frame -- See Shell Window')



cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

vidwriter = cv2.VideoWriter(os.path.splitext(filein)[0]+'_tracked.mp4',cv2.VideoWriter_fourcc('P','I','M','1'), 29.97/2/3, (xmax-xmin,ymax-ymin), True)


k = 0
count = 0
lastcount = 0
centers = []
droplets = []
saved_droplets = []
droplet_count = 0
possible_duplicates = []
while success and k!=27:

    frame = frame[ymin:ymax,xmin:xmax,:]

    
    t = (vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000*14/14.985+t0) ##+dt/2 #because reasons
    #(framerate given as an integer by opencv truncates decimal point
    #       and for some really stupid reasons videos are shot in 29.97fps
    #       instead of 30fps, and this framerate is half that)



    framenum = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    frameBin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameBin = cv2.inRange(frameBin, threshmin,threshmax)
    frameBin = cv2.GaussianBlur(frameBin, (5,5), 0)

    
    if count==lastcount+3:
        #print("-----------------------------------------------")
        #print(f"there are {len(droplets)} droplets")
        lastcount = count


        framedraw = frame.copy()

        result = cv2.findContours(frameBin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if cv2.__version__.startswith('3'):
            conts = np.array(result[1])
        else:
            conts = np.array(result[0])


##        conts = np.array(result[1])

        last_centers = centers
        centers = np.array([findCenter(c) for c in conts])
        boxes = np.array([cv2.boundingRect(c) for c in conts])

        lost_droplets = []
        linked_centers = []
        for i in range(len(droplets)):
            droplet = droplets[i]
##            if droplet.unlinked:
##                print(f"droplet {droplet.dropletid} unlinked for {droplet.unlinked_count} frames")
            if droplet.unlinked_count>=unlinked_count_thresh or droplet.duplicate_count>=duplicate_count_thresh:
                lost_droplets.append(droplet)
                continue
                
            predicted_pos = droplet.filt_pos
            predicted_dims = droplet.filt_dims

            predicted_box = [*(predicted_pos-predicted_dims/2), *(predicted_dims)]

            cv2.rectangle(framedraw, (int(predicted_box[0]),int(predicted_box[1])),\
                           (int(predicted_box[0]+predicted_box[2]), int(predicted_box[1]+predicted_box[3])), (255,255,0), 1)



            try:
                center_diffs = np.sum(np.array([xweight,yweight])*(centers-predicted_pos)**2,axis=1)

                
                contourid = np.argmin(center_diffs)
                overlap = overlapArea(predicted_box, boxes[contourid])/np.abs(np.product(predicted_dims))
                #print(f"droplet {droplet.dropletid} overlap score with {contourid}:",overlap)
                #print(f"Predicted box: {predicted_box}, real box: {boxes[contourid]}")
                if overlap>overlapThresh:
                    #print(f"droplet {droplet.dropletid} is close enough to {contourid}!")

                    if contourid in linked_centers:
                        droplet.duplicate=True
                        droplet.duplicate_count+=1
                    else:
                        droplet.duplicate=False
                        droplet.duplicate_count=0                    
                        linked_centers.append(contourid)

                        
                    droplet.update_observation(centers[contourid], boxes[contourid], contourid)

                    
                else:
                    raise ValueError('No overlap between predicted box and found box')

                

                
            except ValueError:
                droplet.update_observation(None, None, None)






        for i in range(len(lost_droplets)):
            if len(lost_droplets[i].filtered_states)>=50:
                saved_droplets.append(lost_droplets[i])
            droplets.remove(lost_droplets[i])




            
        #print("about to make new droplets")
        for j in range(len(centers)):
            if j not in linked_centers:
                new_droplet = Droplet(centers[j], framenum=framenum, t=t, box=boxes[j], dropletid=droplet_count, contourid=j)
                droplet_count+=1
                droplets.append(new_droplet)






        
##        framedraw = cv2.cvtColor(framedraw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(framedraw, conts, -1, (0,255,0), thickness=1)

        [cv2.circle(framedraw, (int(center[0]), int(center[1])), 3, (0,0,255), -1) for center in centers]
        try:
##            [cv2.rectangle(framedraw, (int(droplet.filt_pos[0]-droplet.filt_dims[0]/2), int(droplet.filt_pos[1]-droplet.filt_dims[1]/2)),\
##                           (int(droplet.filt_pos[0]+droplet.filt_dims[0]/2), int(droplet.filt_pos[1]+droplet.filt_dims[1]/2)), (255,255,0), 1) for droplet in droplets]

            [cv2.putText(framedraw, str(droplet.dropletid), (int(droplet.filt_pos[0]), int(droplet.filt_pos[1])),\
                         cv2.FONT_HERSHEY_SIMPLEX, 1/2, (0,255,255), 1) for droplet in droplets]
        except ValueError:
            pass

        for droplet in droplets:

            cv2.polylines(framedraw, [np.array(droplet.filtered_states)[:,[0,2]].astype(int)], False, (0,0,255), 2)

            if droplet.contourid is not None:
                cv2.rectangle(framedraw, (int(boxes[droplet.contourid][0]), int(boxes[droplet.contourid][1])),\
                   (int(boxes[droplet.contourid][0]+boxes[droplet.contourid][2]), int(boxes[droplet.contourid][1]+boxes[droplet.contourid][3])),\
                   (255,0,255), 1)
##            else:
##                print(f"droplet {droplet.dropletid} not linked to contour")
        

##        tstring = str(datetime.timedelta(seconds=t))#sec2str(t)
##        size, baseline = cv2.getTextSize(tstring, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness=2)
##        cv2.putText(framedraw, tstring, (0,frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,255), thickness=3)
##
##

        cv2.imshow('Tracking', framedraw)
        vidwriter.write(framedraw)

        k = cv2.waitKey(1)






##        prevframe = frame

    success, frame = vidcap.read()
    count+=1



cv2.destroyAllWindows()
vidcap.release()
vidwriter.release()


#End droplet tracking
#-------------------------------------------------------------------------------
#Start droplet pruning



interesting_droplets = pick_droplets(saved_droplets.copy())




#End droplet pruning
#-------------------------------------------------------------------------------
#Start droplet analysis


        
        
#average velocity with field off, on, errors on each, electric field when it's on
voffs, vons, voff_errs, von_errs, Vs = analyze_droplets(interesting_droplets)




np.savetxt(filein[:-4]+'_output.csv', np.array([voffs,vons,voff_errs,von_errs,Vs]).T, header='vel_off[m/s],vel_on[m/s],vel_off_err[m/s],vel_on_err[m/s],V[V]',comments='',delimiter=',')
print('')
print('output saved')








        



