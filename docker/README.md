# DYNATREE in Docker

It is possible to work with DYNATREE scripts in a Docker container. The container gives you access to the data and scripts either in Solara environment or in Jupyter notebook.
You can look at [short demo](https://youtu.be/Li-lFeIvkjs).

## Instructions

* Download the scripts. Use `git clone` or download zip file from github.
  ```
  git clone https://github.com/arborist-mendelu/dynatree.git
  ```
* Install docker, docker-compose and make.
  ```
  sudo apt install -y docker.io docker-compose-v2 make
  ```
* Extract the data with inputs (experimental data) and precomputed script outputs. Use for example NAS disc of our department. The data are located
  at `/ERC/Mereni_Babice_zpracovani/` directory, files 
  `data_babice.tar.gz` and `outputs_dynatree.tar.gz`. Extract into `dynatree/data` and `dynatree/outputs`.
  You can use symlinks. Create also symlink `public` to `outputs` if you want to download files from solara. 
  Create also `jupyter` symlink or directory to keep the Jupyter notebooks created inside Docker.
  ```
  cd dynatree
  ln -s /my/directory/data .
  ln -s /my/directory/outputs .
  ln -s outputs public
  mkdir jupyter
  ```
  The directory structure should look like this
  ```
  dynatree
  ├── data -> /my/directory/data
  ├── docker
  ├── jupyter
  ├── outputs -> /my/directory/outputs
  ├── public -> outputs
  └── scripts
  ```
* Run `make solara` or `make jupyter`. 
  ```
  cd docker
  make solara
  ```
  * For the first time the docker image is created and the start takes about 5min. After this the container starts immediatelly. 
  * Read the output of the container to see the links where solara or jupyter are available. 
