import numpy as np


def create_tt_v(tt):

    tt_tones = np.array(tt['tones'], dtype=int)
    tt_triggers = np.array(tt['triggers'], dtype=int)
    tt_condition = np.array(tt['condition'], dtype=int)
    block = tt['block']
    tt_block = [int(block.split('_0')[1]) for block in tt['block']]

    bin_width = 0.005
    min_value = int(tt_triggers.min())  # Get the minimum value of 'spike_time'
    max_value = int(tt_triggers.max())  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width) 


    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)
    interpolated_type_stim = np.zeros(len(bins) - 1)
    interpolated_block_stim = np.zeros(len(bins) - 1)


    for i in range(len(bins) - 1):
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
            interpolated_freq[i] = np.nan
            interpolated_type_stim[i] = np.nan
            interpolated_block_stim[i] = np.nan

    features = {}
    for i, bin in enumerate(bins[:-1]):
            features[bin] = {
                'Played_frequency': interpolated_freq[i],
                'Condition': interpolated_type_stim[i],
                'Block' : interpolated_block_stim[i],
                'Frequency_changes': stimulus_presence[i]
            }  
    return features
