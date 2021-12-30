#DEFINE VARIABLES
basePath=$1 # Base path for sdfs of starting substructures and pharmacophores
typesName=$2 # Desired name (WITHOUT Extension) of the types file, base name for the folders containing gninatypes, and prefix for the gninatypes files
designTask=$3 

# Make directories for .gninatypes files
mkdir ${designTask}/gninatypes/${typesName}_frags
mkdir ${designTask}/gninatypes/${typesName}_pharms

## Convert SD Files to gninatypes
# Pharmacophoric representations
python typer.py ${basePath}_starting_structures.sdf ./${designTask}/gninatypes/${typesName}_pharm/${typesName}
# Starting Fragments 
python typer.py ${basePath}_pharmacophores.sdf ./${designTask}/gninatypes/${typesName}_frags/${typesName}

## Create Types file
python createTypesFile.py ./${designTask}/gninatypes/${typesName}_pharm/${typesName} ./${designTask}/gninatypes/${typesName}_frags/${typesName} ${typesName}

