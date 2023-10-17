# ACM AI Projects Skeleton Code

## Setup

1. [Create a new conda environment.](https://shine-molecule-f84.notion.site/Installing-Conda-and-PyTorch-db0e9b20677d46c09508b95ed600dd78)

2. Install PyTorch. Go to https://pytorch.org/. Then, scroll down to 'Install Pytorch' and select the right configuration for your machine. Lastly, copy the command that is provided and run it when you have the correct Conda environment activated.

3. As you work on the project, you will end up installing many more packages.

## Running the Skeleton Code

### Running the Code Locally

1. Clone / fork this repository.

2. Open a terminal in the root directory of the repository

3. Activate the conda environment you created using the following command:
```
conda activate <NAME OF ENVIRONMENT>
```
   
4. Populate the conda environment using `acmprojects.yml`:
```
conda env update --file acmprojects.yml --prune
```

5. Go to `start_here.pynb`. This is the main file where you're going to run all of your code from.
