class ParseEFGs:
    output_dir = '../dynpy-examples/Quadrupolar/pre-computed-EFGs/' #Path to directory containing trajectory directories which must be named in the format {01..XX}
    engine = 'ADF' #QM Engine that EFG outputs come from. Can be 'ADF','QE','CP2K'
    prefix = 'I-ADF' # Simple prefix for naming output EFG data file
    time_bw_frames = None #Optional parameter to provide the real-time step between the sampled frames. If not present or set to None, will attempt to read from comment in EFG input/output
