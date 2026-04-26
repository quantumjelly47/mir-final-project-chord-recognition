from IPython.display import Audio, display
import librosa
import matplotlib.pyplot as plt
import mir_eval
import mirdata
import numpy as np
import pretty_midi as pm


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
      for i in range(len(segments)-1):
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
  for i in range(len(mtrack_pcp)-1):
   
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
  
