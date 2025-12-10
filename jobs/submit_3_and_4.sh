#!/bin/bash

# Configuration
START=1
END=192
STEP=10

# Loop through batches
for i in $(seq $START $STEP $END); do
    # Calculate the end of this batch
    batch_end=$((i + STEP - 1))
    
    # Cap the end at 192
    if [ $batch_end -gt $END ]; then
        batch_end=$END
    fi
    
    echo "---------------------------------------------------"
    echo "Processing Batch: $i - $batch_end"
    
    # Submit Case 3
    echo "  > Submitting Case 3..."
    qsub -J $i-$batch_end jobs/gadi/run_gadi_case_3.pbs
    
    # Submit Case 4
    echo "  > Submitting Case 4..."
    qsub -J $i-$batch_end jobs/gadi/run_gadi_case_4.pbs
    
    # Sleep to be polite to the scheduler
    sleep 1
done