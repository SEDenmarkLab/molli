# will generate aso description for conformer library through command line
# optional arguments for feature selection of high variance and low correlation features
# threshold defaults of 0 and 0.8, respectively
import argparse
import alt_aso as aa
import post_processing as pc
import pandas as pd
import h5py                                                                     # DELETE THIS FILE

parser = argparse.ArgumentParser("molli aso",
                                 description="Generates aso descriptor data for a conformer library, with optional post-processing")

parser.add_argument("--file", '-f', action="store", required=True, type=str, metavar='<fpath>', help="ConformerLibrary file to be parsed")

parser.add_argument('--output', '-o', action='store', type=str, default=None, metavar='<fpath>',
                    help='File path for directory to write to. Defaults to "aso.h5" in same directory as alt_aso.py')


args = parser.parse_args()
clib = args.file
output = args.output

aa.aso_description(clib, output)
df = pc.unpack_h5py(output)
print(pc.calc_total_variance(df))

if output is not None:
    temp = output[::-1]                 # reverses file output, returns output without .h5, inserts _post_processed.h5 at end, reverses back
    split = temp.split('.', 1)[1]
    name = '5h.dessecorp_tsop_' + split
    name = name[::-1]
else: 
    name = 'post_processed.h5'

# issues with creating post processed h5 file. maybe move functionality to data_processing.py, so everything is done within loaded dataframe?
#with h5py.File(name, 'w') as f:   # ADD PARALLEL PROCESSING
#    for i in df.index.to_list():
#        f.create_dataset(i, i[:], dtype='f4')
