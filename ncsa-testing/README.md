# Running molli in Docker / Kubernetes
A set of tools and examples for running molli programmatically within the container

## Getting Started
You will need:
* Docker Desktop ([Windows](https://docs.docker.com/desktop/install/windows-install/)/[Mac](https://docs.docker.com/desktop/install/mac-install/)/[Linux](https://docs.docker.com/desktop/install/linux-install/))
* Create a [GitHub Personal Access Token](https://github.com/settings/tokens)

Login to ghcr.io using GitHub PAT:
```
$ docker login ghcr.io
Username: <GITHUB_USERNAME>
Password: <GITHUB_PERSONAL_ACCESS_TOKEN>
Login Succeeded!
```

Then pull the image from a GitHub org that you can access:
If you have access to https://github.com/SEDenmarkLab:
```
$ docker pull ghcr.io/SEDenmarkLab/molli:ncsa-workflow
```

If you have access to https://github.com/moleculemaker:
```
$ docker compose pull
```
which is a shorthand for:
```
$ docker pull ghcr.io/moleculemaker/molli:ncsa-workflow
```

## Run image (default command: batch mode)
To run a new container with the default command:
```
$ docker compose run molli
```
which is a shorthand for:
```
$ docker run -it --rm ghcr.io/moleculemaker/molli:ncsa-workflow
```

The image will be pulled automatically if it is not present.

If the image is not present and can't be pulled, Docker will attempt to build it from source.

## Run image (Jupyter: interactive mode)
Pass `jupyter` as command to the container to override the default command:
```
$ docker compose up
```
which is a shorthand for:
```
$ docker run -it --rm -p 8888:8888 ghcr.io/moleculemaker/molli:ncsa-workflow jupyter
```

## Build image (optional)
To build the image, you can run:
```
$ docker compose build
```
which is a shorthand for:
```
$ docker build -t ghcr.io/moleculemaker/molli:ncsa-workflow .
```


## Run in Kubernetes cluster
In Docker Desktop, choose "Enable Kubernetes" in your Settings and wait for your new cluster to initialize
```
$ kubectl get nodes
$ kubectl get pods -A -w
```

Add private git credentials on Kubernetes cluster
```
$ kubectl create secret docker-registry molliregcred \
            --docker-server=ghcr.io \
            --docker-username=<GIT_USERNAME> \
            --docker-password=<GIT_PERSONAL_ACCESS_TOKEN>
```

Run molli as a Kubernetes Job:
```
$ kubectl apply -f ncsa-testing/molli.job.yaml
job.batch/molli created
```

Watch for status changes of running container(s):
```
$ kubectl get pods -w
NAME          READY   STATUS              RESTARTS   AGE
molli-t5d6v   0/1     ContainerCreating   0          2s
molli-t5d6v   1/1     Running             0          7s
molli-t5d6v   0/1     Completed           0          6m59s
```

View logs of running Job:
```
$ kubectl logs -f job/molli
2023-06-16 17:40:18,713 [INFO] === Starting Job ===
2023-06-16 17:40:18,713 [INFO] === Parsing ChemDraw Files ===
100%|██████████| 5/5 [00:00<00:00, 81.84it/s]
100%|██████████| 15/15 [00:00<00:00, 259.87it/s]
2023-06-16 17:40:19,541 [INFO] === Performing Combinatorial Expansion ===
Will create a library of size 75
100%|██████████| 75/75 [00:31<00:00,  2.40it/s]
2023-06-16 17:40:51,273 [INFO] === Generating Conformers ===
100%|██████████| 75/75 [06:05<00:00,  4.87s/it]
2023-06-16 17:46:57,600 [INFO] 1654 conformers in library
2023-06-16 17:46:57,603 [INFO] === Generating ASO Descriptor ===
100%|██████████| 75/75 [00:00<00:00, 3516.78it/s]
(6688, 3)
Allocating storage for descriptors
Will compute descriptor ASO using 2 cores.
Grid shape: (6688, 3)
Loading batches of conformers: 1it [00:02,  2.92s/it]
2023-06-16 17:47:01,361 [INFO] === Running Post-Processing === 
100%|██████████| 75/75 [00:00<00:00, 3905.65it/s]                    
shape of data after variance threshold: (75, 3490)
total variance after variance threshold: 70.24

shape of data after removing correlated columns (R > 0.8): (75, 502)
total variance after removing correlated columns (R > 0.8): 8.34

2023-06-16 17:47:09,421 [INFO] === Job Complete in 410.7083191871643 seconds ===
```


