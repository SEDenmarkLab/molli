Classifications for Types of Atoms 
===================================

This section contains all the classifications for the types of atoms that are available. Each classification is stored as a class with the same name. 
The attributes show the possible assignments that can be made to the atom.

AtomType:
---------
    Unknown = 0
    
    Regular = 1
   
    Aromatic = 2
   
    CoordinationCenter = 10
   
    Hypervalent = 20

    **Main Group Types**
    
    MainGroup_sp3 = 31
    
    MainGroup_sp2 = 32
    
    MainGroup_sp = 33

    **Non-atom placeholders**
    
    Dummy = 100
    
    AttachmentPoint = 101
    
    LonePair = 102

    **Specific atom classes**
    
    C_Guanidinium = 201
    
    N_Amide = 202
    
    N_Nitro = 203
    
    N_Ammonium = 204
    
    O_Sulfoxide = 205
    
    O_Sulfone = 206
    
    O_Carboxylate = 207

AtomStereo
----------
    Unknown = 0
    
    NotStereogenic = 1

    R = 10
   
    S = 11

    Delta = 20
   
    Lambda = 21

AtomGeom
--------
    Unknown = 0
    
    R1 = 10

    R2 = 20
  
    R2_Linear = 21
    
    R2_Bent = 22

    R3 = 30
   
    R3_Planar = 31
    
    R3_Pyramidal = 32
   
    R3_TShape = 33

    R4 = 40
    
    R4_Tetrahedral = 41
    
    R4_SquarePlanar = 42
    
    R4_Seesaw = 43

    R5 = 50
    
    R5_TrigonalBipyramidal = 51
    
    R5_SquarePyramid = 52

    R6 = 60
    
    R6_Octahedral = 61

