from IPython.display import Audio, display
import librosa
import matplotlib.pyplot as plt
import mir_eval
import mirdata
import numpy as np
import pretty_midi as pm
import pandas as pd
import os


def load_data(dataset_name, data_home, dataset_version="default"):
    """
    Load a specific version of a dataset using the mirdata library.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load, e.g., "gtzan_genre".

    dataset_version : str
        Version of the dataset to load. To load the "mini" version, specify "mini". Default is "1.0".

    data_home : str
        Path to where the dataset is located.

    Returns
    -------
    dataset : mirdata.Dataset
        The initialized mirdata Dataset object corresponding to the specified dataset and version.

    Notes
    -----
    The function is optimized for GTZAN-genre dataset but can be potentially used for other datasets supported by mirdata.
    """
    # Initialize the mini-Medley-Solos-DB dataset at the given location
    dataset = mirdata.initialize(dataset_name, data_home, dataset_version)
    # Return initialized dataset
    return dataset

def beat_times(data, division='downbeat'):
    """
    Uses Pretty_Midi get_downbeats and get_beats to get a dictionary of ids and beat times.

    Parameters
    ----------
        data (dict): Dictionary of multitracks with track_id as the key and multitrack objects as the value.
        division (string): Level of division, either bar (downbeat) or beat

    Returns
    ----------
        beat_times (dict): Dictionary of beat times, track_id as the key and np.ndarray of time values
    """
    # Initialize beat time dictionary
    beat_times = {}

    # Downbeat
    if division == 'downbeat':
      # Iterate over dataset dictionary
      for id, mtrack in data.items():
        # Set downbeat times as the value to the id in segments dictionary
        beat_times[id] = mtrack.midi.get_downbeats()
    # Beats
    elif division == 'beat':
      # Iterate over dataset dictionary
      for id, mtrack in data.items():
          # Set downbeat times as the value to the id in segments dictionary
          beat_times[id] = mtrack.midi.get_beats()

    return beat_times

def segment_midi(data, beat_times):
    """
    Segment multitracks based on beat times, store segment times and active notes in a dictionary

    Parameters
    -----------
        data (dict): Dictionary of tracks with mtrack_id as the key and mtrack objects as the value.
        beat_times (dict): Dictionary of beat times, mtrack_id as the key and np.ndarray of time values

    Returns
    -----------
        segment_dict (dict): mtrack_id as key, an array of 3-key dictionaries storing segment information as value 
    """
    # Intialize segment dictionary
    segment_dict = {}

    # Iterate over all tracks in the beats dictionary
    for id, mtrack in data.items():
    # Create array of all notes for a track
      notes = []
      # Add all notes (PrettyMIDI container) which do not belong to drums.
      for instr in mtrack.midi.instruments:
        if not instr.is_drum:
          notes.extend(instr.notes)

    # Create array of time windows and access beat times for track
      segments = []
      times = beat_times[id]

      for i in range(len(times)-1):

          # Segment start and end is between two elements in beat_times
          start_time = times[i]
          end_time = times[i+1]
          # Find active notes during segment by accessing notes array
          # Only count notes which begin during the segment
          active_notes = [n for n in notes if n.start < end_time and n.end > start_time]

          # Append a dictionary for each segment, storing start/end/notes
          segments.append({
              'start':start_time,
              'end':end_time,
              'notes':active_notes
          })
      # Add segments array to full dictionary for all tracks
      segment_dict[id] = segments
    
    return segment_dict

def weighted_pitch_class(segment_dict):
    """ 
    Gets the weighted pitch class scores per segment of a track

    Parameters
    ----------
        segment_dict (dict): Segment information for each track with mtrack_id as key
        
    Returns
    -------
        pcp_dict (dict): Pitch class profiles for every segment in a track, with mtrack_id as key

    """
    pcp_dict = {}

    for id, segments in segment_dict.items():
    # Scores for the full track
      mtrack_scores = np.zeros_like(segments)
      # Iterate over segments, take one segment at a time
      for i in range(len(segments)):
        segment = segments[i]
        # Create pitch class profile array
        pcp = np.zeros(12)
        # Find duration of segment, return 
        segment_duration = segment['end'] - segment['start']
        #if segment_duration > 0: # to avoid division by 0 **WASNT SURE HOW TO INCORPORATE**
          #return pcp
        
        # For all notes in the segment
        for note in segment['notes']:
          # Calculate duration, including cases where note overlaps with segment division
          note_start = max(note.start, segment['start'])
          note_end = min(note.end, segment['end'])
          note_duration = max(0, note_end - note_start)

          # Add score to pitch class, flatten octaves (according to CASSETTE method)
          pcp[note.pitch % 12] += (note.velocity * note_duration / segment_duration)

        # Normalize scores (according to CASSETTE method)
        total_weight = pcp.sum()
        if total_weight > 0:
            pcp_norm = pcp / total_weight
        else: pcp_norm = pcp

        # Store pitch class profile for segment in array of scores for the track
        mtrack_scores[i] = pcp_norm

      # Store full array of scores in a dictionary for all tracks
      pcp_dict[id] = mtrack_scores
  
    return pcp_dict
    
def get_all_templates(patterns, N = True):
  """
  Given patterns for chord qualities in C, finds full list of templates for all roots. Automatically adds N (no chord) template
  **At the moment, just MAJ and MIN**

    Parameters
    ----------
        patterns (dict): Arrays length 12 corresponding to chord qualities in C, keyed by name of quality
        N (bool): Determines whether to include a No Chord template (default True)
    Returns
    -------
        all_templates (dict): 

  """
  # Dictionary for all templates
  all_templates = {}
  # Names of roots
  names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  # Iterate over all given patterns
  for qual, pattern in patterns.items():
      for root in range(12):
        # Move the pattern by the distance from C
        template = np.roll(pattern, root)
        # Create key name value for templates dictionary and store value
        key_name = f'{names[root]}:{qual}'
        all_templates[key_name] = np.array(template)
  
  # Add No Chord template
  if N:
    all_templates['N'] = np.zeros(12)
  
  return all_templates

def calc_similarity(template, pcp):
  """

  """
  positive = pcp[template == 1].sum()
  negative = pcp[template == 0].sum()
  misses = np.sum((template == 1) & (pcp < 1e-9))
  
  return positive - (negative + misses)

def classify_chord(mtrack_pcp, templates):
  """
    Estimates the chord against all templates based on pitch class profile scores
 
    Parameters
    ----------
    mtrack_pcp (array): Pitch class profile scores for a multitrack
    templates (dict): All templates to compare, keyed by root and quality

    Returns
    -------
    mtrack_chords (array): Array of chord estimations, by segment, for multitrack
    mtrack_sim (array): Array of corresponding best similarity scores, by segment, for multitrack
      
  """
  # Create empty return arrays for chords and similarity scores
  mtrack_chords = [''] * len(mtrack_pcp) # Same len as outer size of pcp scores
  mtrack_sim = np.zeros(len(mtrack_pcp)) 

  # Iterate over length of track pcp scores
  for i in range(len(mtrack_pcp)):
   
    pcp = mtrack_pcp[i] # Current pitch class profile
    best_chord = None    # Base values
    best_root_idx = -1
    best_sim = -float('inf')

    # Calculate similarity against all templates
    for chord, template in templates.items():
        sim = calc_similarity(template, pcp)
        root_idx = np.argmax(template != 0) # Store root of template

        # Update best value if similarity is highest
        if sim > best_sim:
            best_sim = sim
            best_chord = chord
            best_root_idx = np.argmax(template != 0) # Root of best template match

        # Same similarity, choose profile with highest weight at root (CASSETTE)
        elif (sim == best_sim):
            if (pcp[root_idx] > pcp[best_root_idx]):
                best_sim = sim
                best_chord = chord
                best_root_idx = np.argmax(template != 0) 

    # If the best similarity is below -3, give no chord
    if best_sim <= -3:
      best_chord = 'N', best_sim #??Should we return a diff sim value

    # Store values in output array
    mtrack_chords[i] = best_chord
    mtrack_sim[i] = best_sim

  return mtrack_chords, mtrack_sim

def save_cassette_csv(beats, chord_estimates, track, downbeat=False):
  """
  Save CASSETTE model outputs in correct format (start_time, end_time, value)

  Parameters
  ----------
  beats (dict): dictionary of beats keyed by track id
  chord_estimates (dict): dictionary of chord estimates keyed by track id
  track (str): track id
  downbeat (bool): whether saving downbeats (default False)

  Returns
  -------
  None
  """

  cassette_csv = pd.DataFrame({'start_time': beats[track][:-1], 
                                'end_time': beats[track][1:], 
                                'value': chord_estimates[track][0]})
  
  if downbeat:
    os.makedirs('./output/cassette/downbeats', exist_ok=True)
    file_path = f"./output/cassette/downbeats/{track}.csv"
    cassette_csv.to_csv(file_path, index=False)
  else:
    os.makedirs('./output/cassette/beats', exist_ok=True)
    file_path = f"./output/cassette/beats/{track}.csv"
    cassette_csv.to_csv(file_path, index=False)

def majority_label(beat_start, beat_end, segments):
  """
  Finds chord label that occupies majority of interval length

  """
  scores = {}

  for row in segments.itertuples(index=False):
      # compute overlap
      overlap = min(row.end_time, beat_end) - max(row.start_time, beat_start)

      if overlap <= 0:
          continue

      scores[row.value] = scores.get(row.value, 0.0) + overlap

  if not scores:
      return 'N'

  return max(scores, key=scores.get)

def process_crema_df(df, enharmonic_map):
  """
  Process output of crema for easy use with beatwise comparison and chord.evaluate comparison

  Parameters
  ----------
  df (pandas DataFrame): crema df
  enharmonic_map (dict): enharmonic maps for standardization

  Returns
  -------
  df (pandas DataFrame): preprocessed crema df
  """

  df['end_time'] = df['time'] + df['duration']
  df = df.rename(columns={'time': 'start_time'})
  df['value'] = (df['value']
                 .replace({'X': 'N'})
                 .map(lambda x: enharmonic_map.get(x, x)))
  
  return df

def get_beatwise_crema(df, ref_df, majority_label):
  """
  Process crema df for accurate comparison with cassette df beatwise
  """
  crema_beatwise_labels = []

  for i in range(len(ref_df)):
      # get interval
      start = ref_df.loc[i, 'start_time']
      end = ref_df.loc[i, 'end_time']
      # get crema's chord label(s) that are active within the interval
      crema_segments = df[(df['start_time'] < end) & (df['end_time'] > start)][['start_time', 'end_time', 'value']]
      # get majority label
      label = majority_label(start, end, crema_segments)
      crema_beatwise_labels.append(label)

  crema_beatwise_df = pd.DataFrame({'start_time': ref_df['start_time'].values, 
                                      'end_time': ref_df['end_time'].values, 
                                      'value': crema_beatwise_labels})
  
  return crema_beatwise_df

def normalize_chord_label(chord):
    """
    Preprocess chord labels to standardize across crema and cassette's outputs

    Parameters
    ----------
    chord (str): chord label

    Returns
    -------
    Normalized chord label
    """

    if chord in ['N', 'X'] or 'sus' in chord:
       return 'N'

    # remove inversion
    chord = chord.split('/')[0]

    # split root / quality
    root, qual = chord.split(':')

    # chord reduction
    if qual in ['min7', 'hdim7', 'dim7']:
       return f'{root}:min7'
    if (qual in ['min', 'hdim', 'dim']) or (qual.startswith('min')):
        return f'{root}:min'
    if qual in ['maj7']:
       return f'{root}:maj7'
    if (qual in ['maj', 'aug']) or (qual.startswith('maj')):
        return f'{root}:maj'
    return f'{root}:{qual}'

def manual_chord_evaluation(crema_df, cassette_df):
  """
  Evaluate cassette outputs against crema's

  Returns
  -------
  accuracy_list (list): list of accuracies (number of matches / number of beat intervals) per track
  accuracy_inv_list (list): list of accuracies (number of matches / number of beat intervals) per track ignoring inversions
  accuracy_triad_list (list): list of accuracies (number of matches / number of beat intervals) per track capturing triad harmonies
  """
  # combine crema & cassette labels for comparison
  combined_df = cassette_df.merge(crema_df, on=['start_time', 'end_time'], suffixes=['_cassette', '_crema'])

  accuracy = (combined_df['value_cassette'] == combined_df['value_crema']).mean() * 100
  accuracy_no_inv = (
      (combined_df['value_cassette'] ==
      combined_df['value_crema'].str.replace(r'/.*', '', regex=True))
      .mean()
  ) * 100
  accuracy_triad = (combined_df['value_cassette'].apply(normalize_chord_label) 
                    == combined_df['value_crema'].apply(normalize_chord_label)).mean() * 100

  return accuracy, accuracy_no_inv, accuracy_triad

def get_mir_chord_scores(crema_df, cassette_df, normalize_labels=False):
  """
  Use mir_evalchord.evaluate to get scores for crema vs cassette comparison

  Parameters
  ----------
  crema_df (pandas DataFrame)
  cassette_df (pandas DataFrame)
  normalize_labels (bool): whether to normalize chords or not (default False)

  Returns
  -------
  score: individual scores for given track via mir_eval.chord.evaluate
  """
  # snap crema outputs to prevent overlapping of intervals
  crema_df_copy = crema_df.copy()
  cassette_df_copy = cassette_df.copy()

  # align timestamps
  crema_df_copy['start_time'] = crema_df_copy['start_time'].round(6)
  crema_df_copy['end_time'] = crema_df_copy['end_time'].round(6)

  cassette_df_copy['start_time'] = cassette_df_copy['start_time'].round(6)
  cassette_df_copy['end_time'] = cassette_df_copy['end_time'].round(6)

  if normalize_labels:
     score = mir_eval.chord.evaluate(
        cassette_df_copy[['start_time', 'end_time']].values,
        cassette_df_copy['value'].apply(normalize_chord_label).values,
        crema_df_copy[['start_time', 'end_time']].values,
        crema_df_copy['value'].apply(normalize_chord_label).values
    )
  
  else:
    score = mir_eval.chord.evaluate(
        cassette_df_copy[['start_time', 'end_time']].values,
        cassette_df_copy['value'].values,
        crema_df_copy[['start_time', 'end_time']].values,
        crema_df_copy['value'].values
    )

  return score