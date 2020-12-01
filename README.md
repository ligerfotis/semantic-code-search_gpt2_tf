# Semantic Code Search using GPT2 small model as Code and Query Encoder

#### Based on [CodeSearchNet Challenge](https://github.com/github/CodeSearchNet) code

## Train a Model
### Steps to run with Singularity on TACC

1. If running on TACC clusters, load the Singularity module
      
      module load tacc-singularity
      
2. Clone dir in the work directory of TACC
        
        cd $WORK
        git clone https://github.com/ligerfotis/semantic-code-search_gpt2_tf.git
        cd semantic-code-search_gpt2_tf
        
3. To download dataset (only the first time), run from root directory
      
        script/setup
      
4. Start the container by running
    
        script/console
    
5. Train on part of the Python Dataset

        cd src/
        
        train.py --model gpt2 --testrun ../resources/saved_models/ ../resources/data/python/final/jsonl/train/ ../resources/data/python/final/jsonl/valid/ ../resources/data/python/final/jsonl/test
            
6. Alternatively one can sent the training as a job on TACC after step 3.

        sbatch  run_experiments_gtp2_p100.sh 

Docker images:
* preprocessing: ligerfotis/preprocessing:latest
* main image with gpu support:  ligerfotis/csnet:gpu
