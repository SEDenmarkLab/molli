import argparse                                             # DELETE THIS FILE
import generate_conformers as gc


parser = argparse.ArgumentParser(
    "molli conformer_generation",
    description="Read a molli library and generate a conformers for each molecule"
)

parser.add_argument("--file", '-f', action="store", required=True, type=str, metavar='<fpath>', help="MoleculeLibrary file to be parsed.")

parser.add_argument('--n_confs', '-n', action='store', default=50,
                    help=('Max number of conformers generated per molecule. Defaults to 50. Can be set to any int > 0 or to presets, "default" or "quick"'))

parser.add_argument('--iterations', '-i', action='store', type=int, default=10000, help='Max iterations for rdkit conformer embedding as an int. Defaults to 10000')

group = parser.add_mutually_exclusive_group()
group.add_argument('--threshold', '-t', action='store_true',
                   help=('Boolean for energy threshold for filtering conformers. Default is False. Calling this argument will set threshold to 15.0. Mutually exclusive with threshold_value argument'))

group.add_argument('--threshold_value', '--tv', action='store', type=float, default=15.0,
                   help=('Sets specific value for energy theshold for filtering conformers. Default is 15.0. Mutually exclusive with threshold argument'))

parser.add_argument('--output', '-o', action='store', type=str, default=None, metavar='<fpath>',
                    help='File path for ConformerLibrary to write to. Defaults to conformers.mlib in same directory as generate_conformers.py')

parser.add_argument('--num_processes', '-p', action='store', type=int, default=4, help='Number of worker processes in pool. Defaults to 4')

args = parser.parse_args()

mlib = args.file
n_confs = args.n_confs
iterations = args.iterations
if args.threshold:
    threshold = bool(args.threshold)
elif args.threshold_value:
    threshold = float(args.threshold_value)
output = args.output    # function handles output=None
num_processes = args.num_processes

gc.conformer_generation(mlib, n_confs, iterations, threshold, output, num_processes)