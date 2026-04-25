# lieglass-dsc

All mp4 files are LFS pointers. Make sure to have Git LFS setup correctly before cloning or pulling.

Run: 
# From project root
python -m deception_detection.data.preprocessing.preprocess_resized \
    --manifest Data/manifest_dolos.csv \
    --resized_dir Data/Resized-Grayscale-Audio \
    --feature_dir features \
    --workers 4

conda activate LieGlass
cd Shared/
cd Shared\ Vault/
cd Projects/
cd Dammit/
cd lieglass-dsc/

python -m deception_detection.train \
    --manifest Data/manifest_dolos.csv \
    --feature_dir features | tee output.txt
