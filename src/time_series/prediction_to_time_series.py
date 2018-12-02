import numpy as np
import pandas as pd


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def prediction_to_time_series(prediction_or_path_to_numpy_file, sample_rate):
    """
    Converts the model prediction output to the time series
    representation (Onset, Offset, MidiPitch)
    Could be useful for the following converting from time series to MIDI file

    Args:
        prediction_or_path_to_numpy_file(str or np.ndarray[number_of_frames, number_of_notes]):
        path to the file with stored numpy file with model output(prediction) or this numpy object itself

    Returns:
        pandas.Dataframe: time series song representation
    """

    if type(prediction_or_path_to_numpy_file) is str:
        prediction = np.load(prediction_or_path_to_numpy_file)
    else:
        prediction = prediction_or_path_to_numpy_file

    # transpose predictions to loop over each note records easier
    prediction = prediction.transpose()

    time_series = pd.DataFrame(columns=["Onset", "Offset", "MidiPitch"])

    number_notes = prediction.shape[0]
    number_frames = prediction.shape[1]
    win_len = 512 / sample_rate

    index = 0

    # Aux_Vector of times
    vector_aux = np.arange(1, number_frames + 1) * win_len

    for note in range(number_notes):
        cons = np.where(prediction[note] > 0)
        consecutive_groups_of_indexes = consecutive(cons[0])

        for indexes_group in consecutive_groups_of_indexes:
            start_time = indexes_group[0] * win_len
            end_time = indexes_group[-1] * win_len
            time_series.loc[index] = [start_time, end_time, int(note + 21)]
            index += 1

    return time_series


if __name__ == '__main__':
    prediction_to_time_series("/home/andrew/AI/AICourseProject/prediction.npy", 20000)
