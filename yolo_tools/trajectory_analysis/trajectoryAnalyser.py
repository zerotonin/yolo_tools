# -----------------------------------------------------------------------------
# Script: my_analysis_script.py
# Description: This script performs data analysis on video footage, applying
#              filters and extracting valuable information from the data sets.
# Author: 
# Date: updated on 23 Jan 24
# -----------------------------------------------------------------------------

# ----------------------------[ IMPORTS ]-------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import json
import argparse
from yolo_tools.analysis_file_manager.AnalysisFileManager import AnalysisFileManager

# ----------------------------[ CONSTANTS ]-----------------------------------


# -----------------------------------------------------------------------------

# Note: The constants defined above (FPS, MEDFILT_KERNEL, B, A) are used 
#       throughout the script for signal processing and analysis.
#       - FPS: Determines the frame rate of the video for time-based analysis.
#       - MEDFILT_KERNEL: Kernel size for the median filter used to smooth the data.
#       - B, A: Coefficients for the Butterworth low-pass filter to remove high-frequency noise.
#       - LEFT, RIGHT, NEUTRAL: State numbers explaining where the animal is in simple ints, 
#                               to make their tranisitions unique as a diff of the vector.

class trajectoryAnalyser:

    def __init__(self,fps =10,arena_size = (18,8)) -> None:
        # Frames Per Second of the video
        self.fps = fps
        self.arena_size = arena_size
        # Kernel size for median filter
        self.MEDFILT_KERNEL = 15
        # Coefficients for Butterworth low-pass filter
        self.MEDFILT_B, self.MEDFILT_A = signal.butter(N=3, Wn=0.1 / (self.fps / 2), btype='low')
        # Constants representing direction or state
        self.LEFT = -2       # Represents the 'left' or 'rearing' state.
        self.RIGHT = 1       # Represents the 'right' or 'distractor' state.
        self.NEUTRAL = 0     # Represents a 'neutral' state.
        # pre allocation
        self.fly_tra_imageNorm = None
        self.fly_tra_arenaNorm = None
        self.fly_tra_mm = None
        self.arena_imageNorm = None
        self.arena_midline_imageNorm =None
        self.zone_occupation = None
        self.zone_occupation_normalised = None
        self.decision_dict = dict()
        self.locomotor_dict = dict()


    def get_trace(self,trajectories):
        """
        Calculate the trajectory of a fly within a rectangular arena using detection bounding boxes.

        This function computes the fly's trajectory based on its position inside the arena. The arena's
        position is determined as the median of the bounding boxes over time to minimize digitization noise.
        The fly's position is calculated as the mean value of its bounding box. Optionally, the positions
        can be filtered using a signal filter.

        Parameters:
        - row (int): The index of the row where the fly and arena are located in the `bboxes` array.
        - column (int): The index of the column where the fly and arena are located in the `bboxes` array.
        - bboxes (numpy.ndarray): A 4D array containing bounding box coordinates. The dimensions are
        expected to be [frame, row, column, coordinates], where coordinates are in the order
        [y1, x1, y2, x2, fly_y1, fly_x1, fly_y2, fly_x2].
        - filter (bool, optional): If True, applies a signal filter to the fly's position coordinates.
        Requires global variables `B` and `A` to be defined as filter coefficients. Defaults to False.

        Returns:
        - pos_x (numpy.ndarray): The x positions (horizontal) of the fly in each frame, optionally filtered.
        - pos_y (numpy.ndarray): The y positions (vertical) of the fly in each frame, optionally filtered.
        - arena (numpy.ndarray): The median position of the arena's bounding box, reducing digitization noise.
        - arena_midline (float): The midline position of the arena, calculated as the average of y2 and y1 of the arena.

        Note:
        - The arena is assumed to be static, and its position is calculated as the median bounding box over all frames.
        - The fly's position is computed as the midpoint of its bounding box for each frame.
        - Filtering fly positions is optional and can be used to smooth the trajectory.
        """
        self.arena_imageNorm = np.nanmedian(trajectories[:,:4], axis=0)
        self.arena_midline = 0.5 * (self.arena_imageNorm [2] + self.arena_imageNorm [0])

        pos_x = 0.5 * (trajectories[:, 4] + trajectories[:, 6])
        pos_y = 0.5 * (trajectories[:, 5] + trajectories[:, 7])
        self.fly_tra_imageNorm = np.vstack((pos_x,pos_y)).T

    
    def filter_trajectory(self,trajectory):
        trajectory_filtered = trajectory
        trajectory_filtered[:,0] = signal.filtfilt(self.MEDFILT_B, self.MEDFILT_A, trajectory[:,0])
        trajectory_filtered[:,1] = signal.filtfilt(self.MEDFILT_B, self.MEDFILT_A, trajectory[:,1])
        return trajectory_filtered

        

    def get_arena_centered_fly_trace(self):
        self.fly_tra_arenaNorm = np.zeros(shape= self.fly_tra_imageNorm.shape)
        arena_median_pos = np.nanmedian(self.arena_imageNorm)
        self.fly_tra_arenaNorm[:,0] = (self.fly_tra_imageNorm[:,0] - self.arena_imageNorm[0]) / (self.arena_imageNorm[2] - self.arena_imageNorm[0])
        self.fly_tra_arenaNorm[:,1] = (self.fly_tra_imageNorm[:,1] - self.arena_imageNorm[1]) / (self.arena_imageNorm[3] - self.arena_imageNorm[1])


    def get_fly_trace_mm(self):
        self.fly_tra_mm = np.zeros(shape= self.fly_tra_arenaNorm.shape)
        self.fly_tra_mm[:,0] = self.fly_tra_imageNorm[:,0] * self.arena_size[0]
        self.fly_tra_mm[:,1] = self.fly_tra_imageNorm[:,1] * self.arena_size[1] 

    def interpolate_and_fill(self,arr):
        # Indices of the valid values (not np.nan)
        valid_idx = np.where(~np.isnan(arr))[0]
        # Valid values themselves
        valid_values = arr[valid_idx]
        
        # Indices of the full array (for interpolation)
        full_idx = np.arange(arr.size)
        
        # Interpolate using valid values
        interpolated = np.interp(full_idx, valid_idx, valid_values)
        
        # Fill np.nan at the start with the first valid value
        if valid_idx[0] > 0:
            interpolated[:valid_idx[0]] = valid_values[0]
        # Fill np.nan at the end with the last valid value
        if valid_idx[-1] < arr.size - 1:
            interpolated[valid_idx[-1]+1:] = valid_values[-1]
        
        return interpolated

    def interpolate_and_fill_2d(self, matrix, axis=0):
        # Apply the interpolation along the specified axis (0 for columns, 1 for rows)
        if axis == 0:  # For each column
            for i in range(matrix.shape[1]):
                matrix[:, i] = self.interpolate_and_fill(matrix[:, i])
        elif axis == 1:  # For each row
            for i in range(matrix.shape[0]):
                matrix[i, :] = self.interpolate_and_fill(matrix[i, :])
        else:
            raise ValueError("Axis must be 0 or 1.")
        
        return matrix


    def get_side(self,flyX, neutralZone):
        """
        Determine the side of the arena a fly is on based on its X-coordinate.

        Args:
        flyX (float): The X-coordinate of the fly.
        neutralZone (tuple): A tuple representing the start and end of the neutral zone.

        Returns:
        int: A value indicating the side of the arena the fly is on.
            LEFT (-2) if the fly is on the left side,
            RIGHT (1) if the fly is on the right side,
            NEUTRAL (0) if the fly is in the neutral zone.

        The function also defines distinct numbers for transitions between zones,
        calculated as the first derivative of the zone number.

        NEUTRAL  0 -> RIGHT   1:    1  to   right / negative
        RIGHT    1 -> NEUTRAL 0:   -1  from right / negative
        NEUTRAL  0 -> LEFT   -2:   -2  to   left  / positive
        LEFT    -2 -> NEUTRAL 0:    2  from left  / positive
        LEFT    -2 -> RIGHT   1:    3  from left(positive) to right(negative)
        RIGHT    1 -> LEFT   -2:   -3  from right(negative) to left(positive)
        """
        
        if flyX < neutralZone[0]:
            return self.LEFT
        elif flyX > neutralZone[1]:
            return self.RIGHT
        else:
            return self.NEUTRAL
            


    def detect_zone_occupation(self, midline_boundaries, positive_stimulus_is_on_the_left):
        """
        Analyze the trace of the fly movement and determine the side of the midline the fly is on.

        Parameters:
        trace (numpy.ndarray): Array of fly positions, where each position is represented as [x, y].
        positive_stimulus_is_on_the_left (bool): Ture if the positive stimulus  is on the left.
        midline_percentage (float, optional): The percentage of the midline width relative to the total size.
                                            Defaults to 0, meaning no neutral zone.

        Returns:
        numpy.ndarray: Array indicating the side of the midline for each position in trace.
        numpy.ndarray: Array indicating the side of the midline considering the rearing direction.
        
        Note:
        The fly is in a coordinate system where the height and width are normalized to 1.
        A width coordinate below 0.5 indicates the fly is on the left side, and above 0.5 indicates the right side.
        """

        if type(midline_boundaries) == float:
            # position of midline as we are in arena normed coordinates
            midline = 0.5  
        
            # midline width = midline_percentage of total size up and down
            midline_left  = midline - midline_boundaries* 0.5
            midline_right = midline + midline_boundaries* 0.5
        elif type(midline_boundaries) == tuple:
            midline_left  = midline_boundaries[0]
            midline_right = midline_boundaries[1]
        else:
            raise ValueError(f'midline boundaries are not correctly defined: {midline_boundaries}')


        self.zone_occupation = np.array([self.get_side(self.fly_tra_arenaNorm[i,0], neutralZone=[midline_left, midline_right]) for i in range(self.fly_tra_arenaNorm.shape[0])])
        
        # Call swap_decisions to ensure consistency in decision data
        self.zone_occupation_normalised = self.swap_decisions(self.zone_occupation, positive_stimulus_is_on_the_left)
        
        return 

    def swap_decisions(self,side,positive_stimulus_is_on_the_left):
        """
        Adjust the side decisions based on the rearing side of the fruit fly.
        
        This function swaps the 'side' values if the food the fly was reared on is not on the left.
        This standardizes the decision data, ensuring that the analysis is consistent 
        regardless of the actual side the reared food source was on. The function swaps
        -2 with 1 and keeps other values unchanged.
        
        Args:
            side (numpy.ndarray): Array of side decisions.
            positive_stimulus_is_on_the_left (bool): True if the positive (+1 in preference index) is on the left
                                If False, the data will be flipped to standardize the analysis.
        
        Returns:
            numpy.ndarray: The adjusted array of side decisions.
        """
        if positive_stimulus_is_on_the_left == False:
            # Swap decision values to standardize data based on rearing side
            side = np.where(side == self.RIGHT, 42, side)
            side = np.where(side == self.LEFT, self.RIGHT, side)
            side = np.where(side == 42, self.LEFT, side)
        
        return side


    def decision_analysis(self):
        """
        Analyze the decisions made by an animal based on the given trace data.

        Args:
        - side (ndarray): Array representing the side chosen by the animal (-2 for left, 1 for right, 0 for middle).
        - rear_side (ndarray): Array representing rearing or distraction (-2 for rearing, 1 for distraction).
        - trace (ndarray): Trace data of the animal's position or state over time.

        Returns:
        - time_left (float): Proportion of time spent on the left side.
        - time_middle (float): Proportion of time spent in the middle.
        - time_right (float): Proportion of time spent on the right side.
        - time_rearing (float): Proportion of time spent rearing.
        - time_distract (float): Proportion of time spent being distracted.
        - transition_times (ndarray): Indices in the trace where a transition occurs.
        - transition_directions (ndarray): Directions of the transitions at the transition times.
        - transition_durations (ndarray): Duration of each transition.
        """
        # Get fractions due to position
        self.decision_dict['fraction_left']   = np.count_nonzero(self.zone_occupation == self.LEFT) / self.fly_tra_arenaNorm.shape[0]
        self.decision_dict['fraction_right']  = np.count_nonzero(self.zone_occupation == self.RIGHT)  / self.fly_tra_arenaNorm.shape[0]
        self.decision_dict['fraction_middle'] = np.count_nonzero(self.zone_occupation == self.NEUTRAL)  / self.fly_tra_arenaNorm.shape[0]
        # Get fractions based on stimulus
        self.decision_dict['fraction_positive'] = np.count_nonzero(self.zone_occupation_normalised == self.LEFT)  / self.fly_tra_arenaNorm.shape[0]
        self.decision_dict['fraction_negative'] = np.count_nonzero(self.zone_occupation_normalised == self.RIGHT) / self.fly_tra_arenaNorm.shape[0]

        # The first derivative of positions indicates a decision 
        self.decision_dict['transitions'] = np.append(0, np.diff(self.zone_occupation_normalised))
        self.decision_dict['transition_times'] = np.nonzero(self.decision_dict['transitions'])[0]
        self.decision_dict['transition_directions'] = self.decision_dict['transitions'][self.decision_dict['transition_times']]
        self.decision_dict['transition_durations'] = np.diff(np.append(self.decision_dict['transition_times'], self.fly_tra_arenaNorm.shape[0]))    
        self.decision_dict['time_of_first_decision_elapsed_sec'] = self.decision_dict['transition_times'][0]/self.fps if len(self.decision_dict['transition_times']) >0 else None
        self._collate_decisions()

        #caculate preference indices
        self.decision_dict['preference_index'] =  self.calc_Michelson_contrast(self.decision_dict['fraction_positive'],self.decision_dict['fraction_negative'])
        self.decision_dict['decision_duration_index']  = self.calc_Michelson_contrast(self.decision_dict['decision_duration_matrix'][0,0],self.decision_dict['decision_duration_matrix'][0,1])
        self.decision_dict['time_decision_record'] = np.vstack((self.decision_dict['transition_times'],self.decision_dict['transition_directions'])).T

    def _collate_decisions(self):
        """
        Analyze transitions and their durations to collate decision metrics.

        This function processes a series of state transitions and their respective durations to compute
        the frequency of transitions between specific states and the cumulative duration of these transitions.
        The states are coded as follows:
        - 0: middle
        - 1: negative
        - -2: positive

        Transition differences are explained as:
        -  0 ->  1:  1 (to distractor)
        -  1 ->  0: -1 (from distractor)
        -  0 -> -2: -2 (to rearing)
        - -2 ->  0:  2 (from rearing)
        - -2 ->  1:  3 (from rearing to distractor)
        -  1 -> -2: -3 (from distractor to rearing)

        Parameters:
        - transDirec (list or array): A sequence of integers representing the direction of transitions.
        - transDur (list or array): A sequence of numbers representing the duration of each transition.

        Returns:
        - decisions (numpy array): A 2x2 matrix where the elements represent the count of specific transitions:
            - decisions[0,0]: Count of transitions to rearing
            - decisions[0,1]: Count of transitions to distractor
            - decisions[1,0]: Count of transitions from rearing
            - decisions[1,1]: Count of transitions from distractor

                          toPositive | toNegative       0,0 | 0,1
            decisions = ----------------------------   -----------
                        fromPositive | fromNegative     1,0 | 1,1
        
        - descDur (numpy array): A 1x2 matrix representing the cumulative duration of specific states:
            - descDur[0,0]: Total duration in rearing state
            - descDur[0,1]: Total duration in distractor state

            descDur =[ rearing , distractor]
        
        """
        decisions = np.zeros(shape=(2,2))
        descDur   = np.zeros(shape=(1,2))
        for decI in range(len(self.decision_dict['transition_directions'])):
            if self.decision_dict['transition_directions'][decI] == -1:
                decisions[1,1] +=1
            elif self.decision_dict['transition_directions'][decI] == 1:
                decisions[0,1] +=1
                descDur[0,1]+= self.decision_dict['transition_durations'][decI]
            elif self.decision_dict['transition_directions'][decI] == 2:
                decisions[1,0] +=1
            elif self.decision_dict['transition_directions'][decI] == -2:
                decisions[0,0] +=1
                descDur[0,0]+= self.decision_dict['transition_durations'][decI]
            elif self.decision_dict['transition_directions'][decI] == 3:
                decisions[1,0] +=1
                decisions[0,1] +=1
                descDur[0,1]+= self.decision_dict['transition_durations'][decI]
            elif self.decision_dict['transition_directions'][decI] == -3:
                decisions[0,0] +=1
                decisions[1,1] +=1
                descDur[0,0]+= self.decision_dict['transition_durations'][decI]
        
        self.decision_dict['decision_four_field_matrix'] = decisions
        self.decision_dict['decision_duration_matrix'] = descDur

    def calc_Michelson_contrast(self,a,b):
        """"
        Calculate the Michelson contrast for two values.

        This function computes the Michelson contrast, which is a measure used in various scientific fields. 
        In behavioral science, this contrast is identical to what is called a 'score'. The Michelson contrast 
        is calculated as (a - b) / (a + b) when the sum of a and b is greater than 0. If a + b is not 
        greater than 0, the function returns NaN to indicate that the calculation cannot be performed.

        A value of 1 represents that all of the data is a. A value of 0 represents an eauql distribution of a 
        and b, while -1 represents an all b distribution.
        
        Parameters:
            a (float): The first value used in the calculation.
            b (float): The second value used in the calculation.

        Returns:
            float: The Michelson index if a + b is greater than 0; otherwise, np.nan.
        """
        
        if a + b > 0:
            return (a - b) / (a + b)
        else:
            return np.nan
    
    def locomotor_analysis(self):
        self.locomotor_dict['distance_walked'] = np.sqrt(np.diff(self.fly_tra_mm[:,0]) ** 2 + np.diff(self.fly_tra_mm[:,1]) ** 2)
        # speed = ds/dt
        # ds is in distance walked
        # dt = 1/FPS
        # v = ds/dt = FPS * ds
        self.locomotor_dict['max_speed'] = np.max(self.locomotor_dict['distance_walked']) * self.fps  # delta_distance/delta_time
        self.locomotor_dict['distance_walked'] = np.sum(self.locomotor_dict['distance_walked'])
        # total distance/total time
        self.locomotor_dict['avg_speed'] = self.locomotor_dict['distance_walked'] / (self.fly_tra_mm.shape[0]/self.fps)
    

    def analyse_trajectory(self,detections,midline_boundaries,pos_stimulus_on_left,filter_flag = False):
        """
        Analyzes each arena and collects results.
        
        Args:
        - metadata (DataFrame): The metadata for analysis.
        - bboxes (ndarray): The bounding boxes from video analysis.
        
        Returns:
        - list: The list of results from analyzing each arena.
        """


        self.get_trace(detections)
        self.fly_tra_imageNorm = self.interpolate_and_fill_2d(self.fly_tra_imageNorm,axis=0)

        if filter_flag:
            self.fly_tra_imageNorm = self.filter_trajectory(self.fly_tra_imageNorm)
        
        self.get_arena_centered_fly_trace()
        self.get_fly_trace_mm()
        self.detect_zone_occupation(midline_boundaries,pos_stimulus_on_left)
        self.decision_analysis()
        self.locomotor_analysis()
                




    def plot_trace(pos_x,pos_y,arena,arena_midline,title=None,transitions=None):
        """
        Plot the trace of an object in 2D space over time as recorded in video frames.

        This function creates a 2D plot of the object's x and y positions over time. It also plots the arena's boundaries and, optionally, 
        the transitions of the object as well as a title for the plot.

        Parameters:
        - pos_x (list or array): The x positions of the object over time.
        - pos_y (list or array): The y positions of the object over time.
        - arena (list): A list containing the coordinates [y_min, y_max, x_min, x_max] of the arena's boundaries.
        - arena_midline (float): The midline value of the arena for plotting.
        - title (str, optional): The title of the plot. Defaults to None.
        - transitions (list or array, optional): The time points (frame indices) where transitions occur. Defaults to None.

        The function will generate a plot with 2 subplots:
        1. The first subplot shows the x position over time with the arena's x boundaries and the midline.
        2. The second subplot shows the y position over time with the arena's y boundaries.

        Transitions are marked on the x-axis plot if provided. The arena boundaries are indicated as horizontal lines, and the object's trace is 
        shown as a blue line. If a title is provided, it is set as the supertitle of the plot.

        Returns:
        None
        """
        fig, ax = plt.subplots(2, 1, sharex=True, dpi=300)
        if title is not None:
            fig.suptitle(title)

        ax[0].plot(pos_x,'b')
        ax[0].plot(arena[2] * np.ones_like(pos_x))
        ax[0].plot(arena[3] * np.ones_like(pos_x))
        ax[0].plot(arena_midline * np.ones_like(pos_x),'k-')

        if title is not None:
            ax[0].plot(transitions,np.ones_like(transitions)*arena_midline,'r*')
        ax[0].set_ylabel('position x-axis (pixels)')

        ax[1].plot(pos_y,'b')
        ax[1].plot(arena[0] * np.ones_like(pos_y))
        ax[1].plot(arena[1] * np.ones_like(pos_y))

        ax[1].set_ylabel('position y-axis (pixels)')
        ax[1].set_xlabel('video frames')

        plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process video trajectories and analyze movements.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file containing trajectory data.')
    parser.add_argument('--midline_tolerance', type=float, required=True, help='Tolerance for midline calculation.')
    parser.add_argument('--positive_stimulus_on_left', type=bool, required=True, help='True if the positive stimulus is on the left.')
    parser.add_argument('--filter_trajectory', type=bool, required=True, help='Apply filter to trajectory data.')
    parser.add_argument('--output_locomotion_file', type=str, required=True, help='Output path for locomotion data in JSON format.')
    parser.add_argument('--output_decision_file', type=str, required=True, help='Output path for decision data in JSON format.')

    args = parser.parse_args()

    # Load trajectories from the specified input file
    trajectories = np.load(args.input_file)

    # Initialize the trajectory analyzer with the midline tolerance as a tuple for boundaries
    traAna = trajectoryAnalyser()
    traAna.analyse_trajectory(trajectories, args.midline_tolerance, args.positive_stimulus_on_left, args.filter_trajectory)


    choice_json_keys = ['fraction_left', 'fraction_right', 'fraction_middle', 'fraction_positive', 'fraction_negative', 'preference_index', 'decision_duration_index','time_of_first_decision_elapsed_sec']
    choice_numpy_keys =['transitions', 'transition_times', 'transition_directions', 'transition_durations', 'time_decision_record', 'decision_four_field_matrix', 'decision_duration_matrix']
    choice_json_dict = {key: traAna.decision_dict.get(key, None) for key in choice_json_keys}
    file_manager = AnalysisFileManager()

    # Write decision dictionary to JSON file
    with open(file_manager.create_result_filepath(args.output_decision_file,'choice_json'), 'w') as outfile:
        json.dump(choice_json_dict, outfile, indent=4)
    
    for name in choice_numpy_keys:
        np.save(file_manager.create_result_filepath(args.output_decision_file,name),traAna.decision_dict[name])


    # Write locomotion dictionary to JSON file
    with open(file_manager.create_result_filepath(args.output_locomotion_file,'locomotor_json'), 'w') as outfile:
        json.dump(traAna.locomotor_dict, outfile, indent=4)

    np.save(file_manager.create_result_filepath(args.output_locomotion_file,'tra_mm'),traAna.fly_tra_mm)

    print("Analysis complete. Data saved to specified files.")

if __name__ == "__main__":
    main()

# trajectories = np.load('./tra.npy')
# traAna = trajectoryAnalyser()
# traAna.analyse_trajectory(trajectories,0.1,True,True)