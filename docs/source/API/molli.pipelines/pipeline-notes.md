# molli.pipeline: External Drivers

This discusses the external drivers and a bit how the `Job` class functions. For example use cases, click [here](../../cookbook/012-jobmap.ipynb)!

## Structure of Jobs

The `molli` `jobmap` function takes an input library, a `Job` to be performed on the members of the library, the output library to be written to, a cache directory to write intermediate outputs, and a scratch directory to run each `Job`. The `Molecule` or `ConformerEnsemble` objects will be serialized in the new objects. Methods that are operate with the `Job` class require the specification of two methods: `prep` and `post`.

- `prep` - operates on a `Molecule` or `ConformerEnsemble` from the library and creates a `JobInput`. This will have various attributes including a Job ID, a list of commands to be run, specifications of output streams, and files to be cached. 
- `post`- takes the method in the driver, redefines the methhod for the final step of the `Job` to process the output file. This will take an encoded output file, attempt to create a `JobOutput`, and execute the new method to create an object with the updated attributes if the `Job` was performed as expected. 

`jobmap` takes all instances of `JobInput`, passes them to a ThreadPoolExecutor, and then splits them based on the number of workers requested. For example, if 4 cores are requested with 4 workers, this will partition the submissions such that 16 cores of the CPU will be used. While CPython typically does not gain performance with thread-based parallelism on CPU-bound tasks, this task can be effectively considered an I/O-bound task since the computing is done by an external process.  Upon completion of a job, the outputs will be collected and encoded in a single .out file.

`jobmap_sge` functions the same as `jobmap`, with the main difference is that it was created for submission of jobs through a scheduler. Different configurations of clusters may require additional specification in the submission, so an additional option for a header in the batch submission is available. This function will monitor Job IDs as they complete on the cluster and still capture respective outputs requested from the Job.

In theory, any function can be submitted to molli maintaining the syntax seen in varying drivers. This was designed to allow varying drivers and interfaces to be implemented

## External Drivers and Available Methods

- ORCA
    - All functions utilize a template to create and format ORCA input files. A parser identifies various properties from the m_orca_property.txt file using regular expressions. This has been tested on ORCA 5.0 and higher. The parser identifies various properties from this file: updated coordinates, SCF Energy or VDW Corrections, Mayer population analysis, MDCI_Energies, solvation details, dipole moments, DFT energy, calculated NMR shifts, calculated Hessians, and thermochemistry values

    - Methods Available:
        - `basic_calc_m` - allows implementation of various routine calculations, including single point energy calculations, geometry optimizations, vibrational frequency calculations, etc.
        - `optimize_ens` - same as `basic_calc_m` but operates on `ConformerEnsemble` insead of a `Molecule`
        - `giao_nmr_m` - allows calculation of NMR shifts for specified elements
        - `giao_nmr_ens` - same as `giao_nmr_m` but operates on a `ConformerEnsemble` insead of a `Molecule`
        - `scan_dihedral` - calculates a potential energy surface scan for a 360&deg; rotation around four atoms of interest. Returns a `ConformerEnsemble` of each step

- CREST
    - These functions support forwarding of CREST command-line parameters, such as the XTB method, temperature, energy window, the length of the metadynamics simulation, the dump frequency at which coordinates are written to the trajectory file, the dump frequency in which coordinates are given to the variable reference structure list, and checking for changes in topology. These functions allow further miscellaneous command specification.

    - Methods Available:
        - `conformer_search` - runs a general conformational search on a `Molecule` and creates a `ConformerEnsemble`
        - `conformer_screen` - runs the screen ensemble optimization protocol in CREST for a ConformerEnsemble. This optimizes points along a trajectory and then sorts the conformer ensemble based on energy, rotational constants, and Cartesian RMSDs with a specified XTB method
- XTB
    - These functions support forwarding of the XTB command line arguments. A custom input file can be specified, although the user is responsible for setting up its contents.

    - Methods Available:
        - `optimize_m` - performs an XTB optimization of a Molecule and returns a new Molecule instance with the optimized coordinates
        - `optimize_ens` - same as `optimize_m` but operates on a `ConformerEnsemble` instead of a `Molecule`
        - `energy_m` - performs an XTB energy calculation and adds this as an attribute to the `Molecule`
        - `scan_dihedral` - calculates a potential energy surface scan for a 360&deg; rotation around four atoms of interest. Returns a `ConformerEnsemble` of each step
        - `atom_properties_m` - runs an XTB energy calculation and includes a parser that identifies calculated properties from the output file. The properties parsed for each atom are dispersion, polarizability, charge, covalent coordination number, three Fukui indices, and Wiberg bond index. These are stored as properties of each Atom in the Molecule returned.


- NWChem
    - This method utilizes an NWChem template to help create and format NWChem input files.
    
    -Methods Available:
        - `optimize_atomic_esp_charges_m` - calculates charges for each atom after calculation of an electrostatic potential. If specified, a DFT optimization can be run before electrostatic potential calculation. In addition, the electrostatic minimum and maximum of the collective grid points can be stored as an attribute of the `Molecule`.