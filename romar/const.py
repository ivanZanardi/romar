# See: https://physics.nist.gov/cuu/Constants
from scipy import constants
from scipy.constants import physical_constants as pc

# Universal constants
# -------------------------------------
UH        = pc["Planck constant"][0]               # [J/Hz]
UHT       = pc["reduced Planck constant"][0]       # [J*s]
UEPS0     = pc["vacuum electric permittivity"][0]  # [F/m]
UMU0      = pc["vacuum mag. permeability"][0]      # [N/A^2]
UC0       = pc["speed of light in vacuum"][0]      # [m/s]

# Physico-chemical constants
# -------------------------------------
UAMU      = pc["atomic mass constant"][0]          # [kg]
UNA       = pc["Avogadro constant"][0]             # [1/mol]
UKB       = pc["Boltzmann constant"][0]            # [J/K]
URG       = pc["molar gas constant"][0]            # [J/(mol*K)]

# Atomic and nuclear constants
# -------------------------------------
UE        = pc["elementary charge"][0]             # [C]
GEM       = pc["electron g factor"][0]             # []
UME       = pc["electron mass"][0]                 # [kg]
UMME      = pc["electron molar mass"][0]           # [kg/mol]
UALPHA    = pc["fine-structure constant"][0]       # []
UA0       = pc["Bohr radius"][0]                   # [m]

# Conversion factors
# -------------------------------------
A_to_m    = constants.angstrom                     # [A]    -> [m]
atm_to_Pa = constants.atm                          # [atm]  -> [Pa]
eV_to_J   = constants.eV                           # [eV]   -> [J]  Electronvolt to Joule
Ha_to_J   = pc["Hartree energy"][0]                # [Ha]   -> [J]  Hartree to Joule
Ha_to_eV  = pc["Hartree energy in eV"][0]          # [Ha]   -> [eV] Hartree to Electronvolt
cm3_to_m3 = 1e-6                                   # [cm^3] -> [m^3]

# Limits
# -------------------------------------
TMIN      = 3e2                                    # Minimum temperature [K]
TMAX      = 1e5                                    # Maximum temperature [K]
XMIN      = 1e-20                                  # Minimum molar fraction