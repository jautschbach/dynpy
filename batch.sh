for i in {01..10}; do 
    sed -i "s/SS/$i/" SR.slm; 
    sbatch SR.slm; 
    sed -i "s/$i/SS/" SR.slm; 
done
