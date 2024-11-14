import numpy as np
<<<<<<< HEAD
fs = 30000

def create_tt_v1(tt):
=======


def create_tt_v(tt):
>>>>>>> b1bfb22d (new tt)

    tt_tones = np.array(tt['tones'], dtype=int)
    tt_triggers = np.array(tt['triggers'], dtype=int)
    tt_condition = np.array(tt['condition'], dtype=int)
    block = tt['block']
    tt_block = [int(block.split('_0')[1]) for block in tt['block']]
<<<<<<< HEAD
    tt_triggers = tt_triggers/fs
=======
>>>>>>> b1bfb22d (new tt)

    bin_width = 0.005
    min_value = int(tt_triggers.min())  # Get the minimum value of 'spike_time'
    max_value = int(tt_triggers.max())  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width) 


    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)
    interpolated_type_stim = np.zeros(len(bins) - 1)
    interpolated_block_stim = np.zeros(len(bins) - 1)


<<<<<<< HEAD
    for i in range(1,len(bins) - 1):
=======
    for i in range(len(bins) - 1):
>>>>>>> b1bfb22d (new tt)
        bin_start = bins[i]
        bin_end = bins[i + 1]

            # Check if any stimuli fall within the current bin
        stimuli_in_bin = (tt_triggers >= bin_start) & (tt_triggers < bin_end)
        if np.any(stimuli_in_bin):
                # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

                # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
                # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = tt_tones[stimuli_in_bin][0]
            interpolated_type_stim[i] = tt_condition[stimuli_in_bin][0]
            interpolated_block_stim[i] = tt_block[stimuli_in_bin][0]
                
        else:
                # If no stimulus in the bin, set bin_frequencies to the previous frequency
<<<<<<< HEAD
            interpolated_freq[i] = interpolated_freq[i-1]
            interpolated_type_stim[i] = interpolated_type_stim[i-1]
            interpolated_block_stim[i] = interpolated_block_stim[i-1]
=======
            interpolated_freq[i] = np.nan
            interpolated_type_stim[i] = np.nan
            interpolated_block_stim[i] = np.nan
>>>>>>> b1bfb22d (new tt)

    features = {}
    for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }  
<<<<<<< HEAD
    return features


def create_tt_v(tt):
    # Convert necessary inputs to numpy arrays for efficient processing
    tt_tones = np.array(tt['tones'], dtype=int)
    tt_triggers = np.array(tt['triggers'], dtype=float)
    tt_condition = np.array(tt['condition'], dtype=int)
    tt_block = [
        int(block.split('_0')[1]) for block in tt['block'] if '_0' in block
    ]
    tt_triggers = tt_triggers/fs

    bin_width = 0.005
    min_value = tt_triggers.min()
    max_value = tt_triggers.max()

    # Initialize the features dictionary
    features = {}
    current_bin_start = min_value

    # Iterate over bins dynamically
    while current_bin_start < max_value:
        bin_end = current_bin_start + bin_width

        # Check stimuli in the current bin
        stimuli_in_bin = (tt_triggers >= current_bin_start) & (tt_triggers < bin_end)

        if np.any(stimuli_in_bin):
            # If stimuli are present, retrieve the first matching values
            idx = np.where(stimuli_in_bin)[0][0]
            played_frequency = tt_tones[idx]
            condition = tt_condition[idx]
            block = tt_block[idx]
            frequency_changes = True
        else:
            # Default values for empty bins
            played_frequency = 0
            condition = 0
            block = 0
            frequency_changes = False

        # Store the computed data in the features dictionary
        features[current_bin_start] = {
            'Played_frequency': played_frequency,
            'Condition': condition,
            'Block': block,
            'Frequency_changes': frequency_changes,
        }

        # Move to the next bin
        current_bin_start = bin_end

    return features


import pickle

session = 'ALTAI_20240822_SESSION_00'
path = '/Volumes/data2/eTheremin/ALTAI/'+ session + '/positions'
folder = '/Volumes/data2/eTheremin/ALTAI/'+ session +'/'

file_path = folder + 'headstage_0/tt.pkl'

with open(file_path, 'rb') as file:
    tt = pickle.load(file)

features = create_tt_v(tt)
with open(folder + 'headstage_0/features.pkl', 'wb') as file:
    pickle.dump(features, file)
=======
    return features
>>>>>>> b1bfb22d (new tt)
