import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import h5py


# --- A little script to preprocess whitened time segments, one file at a time --- #


# --- Please specify the input file and the output file --- #
# --- The input file should be in .hdf5 format, and have to specify the flag in the hdf5 --- #
# --- The output file will be like .npz file, with (n,200) in shape and proper snr data--- #
# --- One can add more function to make it --- #


def Preprocessing(input_file_path, output_file_path, mode, num_of_events, keys_for_extraction = None, keys_for_reference = None, keys_for_snr = None, event_size = 200, location_of_snr = -1):
    
    file = h5py.File(input_file_path, 'r')
    
    if keys_for_extraction == None:
        print("Seems you don't know the hdf5 file keys value. That's fine and I'll print that for you.\n")
        print("The keys for the .hdf5 file is list below:\n")
        print(file.keys())
        return
        
    whitened_strain_data = file[keys_for_extraction]
    
    # --- Time to specify the mode --- #
    # --- If it's in "noise" mode, the function will extract roughly the same number of noise events from the strain data ---#
    # --- If it's in "signal" mode, the function will need a "keys_for_reference" to acquire the merger time and extract event around the time ---#
    # --- If it's in "glitch" mode, the function will just give the central n values in the strain data ---#
    
    if mode == 'noise':
        # --- mode for extracting noise events --- #
        event_selected = np.zeros((num_of_events, event_size))
        
        starting_points = np.floor(np.random.random(num_of_events) * 12288).astype(int) + 6144
        num_of_events_per_segment = num_of_events // len(whitened_strain_data) + 1
        # print(num_of_events_per_segment)
        # --- This split procedure maybe problematic as more events may be extracted from the last segment, Can further improve it --- #
        
        for i in range(num_of_events):
            event_selected[i] = whitened_strain_data[i // num_of_events_per_segment][starting_points[i]:starting_points[i] + 200]
            
        np.save(output_file_path, event_selected) 
        
        print("Noise events selected and saved to " + output_file_path)  
        
        
    elif mode == 'signal':
        # --- mode for extracting signal events --- #
        print("In signal mode, the num_of_events parameter will not working and we'll extract all the signal events. \n")
        
        
        if (keys_for_reference == None) or (keys_for_snr == None):
            print("Seems you didn't specify the keys for reference waveform or snr for the injection, which is important to pick out events around merger time. That's fine and I'll print that for you.\n")
            print("The keys for the .hdf5 file is list below:\n")
            print(file.keys())
            return
        
        num_of_signal_events = len(file[keys_for_extraction])
        
        assert (len(file[keys_for_reference]) == num_of_signal_events) and (len(file[keys_for_snr]) == num_of_signal_events), "The number of segments for injected, reference and the snr is different. The file maybe problematic. "
        
        event_selected = np.zeros((num_of_signal_events, event_size))
        
        # --- Below we extract the LIGO style snr data--- #
        snr_data = np.zeros(0)
        for i in range(len(file[keys_for_snr])):
            snr_data = np.append(snr_data,file[keys_for_snr][i][location_of_snr])
        
        
        reference_strain = np.array(file[keys_for_reference])
        merger_time = np.argmax(reference_strain, axis = 1)
        # --- Here we take when the value of the strain is the largest as the merger time. --- #
        
        # print(merger_time)

        for i in range(num_of_signal_events):
            event_selected[i] = whitened_strain_data[i][(merger_time[i] - event_size // 2):(merger_time[i] + event_size - event_size // 2)]
        
        np.savez(output_file_path, strain_time_data = event_selected, snr_data = snr_data)
        
        print("Signal events selected and saved to " + output_file_path) 
        
    elif mode == 'glitch':
        # --- mode for extracting glitch events --- #
        print("In glitch mode, the num_of_events parameter will not working and we'll extract all the glitch events. \n")
        
        if (keys_for_snr == None):
            print("Seems you didn't specify the keys for snr for the glitch. That's fine and I'll print that for you.\n")
            print("The keys for the .hdf5 file is list below:\n")
            print(file.keys())
            return
        
        num_of_signal_events = len(file[keys_for_extraction])
        
        assert (len(file[keys_for_snr]) == num_of_signal_events), "The number of glitches and the snr is different. The file maybe problematic. "
        
        event_selected = np.zeros((num_of_signal_events, event_size))
        
        snr_data = np.zeros(0)
        for i in range(len(file[keys_for_snr])):
            snr_data = np.append(snr_data,file[keys_for_snr][i][location_of_snr])
            
        # glitch_time = np.argmax(whitened_strain_data, axis = 1)
        
        for i in range(num_of_signal_events):
            event_selected[i] = whitened_strain_data[i][12278:12478]
            
        
        np.savez(output_file_path, strain_time_data = event_selected, snr_data = snr_data)
        
        print("Glitch events selected and saved to " + output_file_path) 
        
    else:
        print("No such mode for the function. Please specify the mode again. \n")