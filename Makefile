# Default results directory:
SN_RESULTS_DIR ?= /data/drive2/sn/docker_results

build:
	sudo docker build \
	-f docker/QueueEnvDockerfile \
	--target base \
	--tag queue_env \
	.

run:
	sudo docker run -it --gpus all \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-v $(shell pwd)/queue_env:/queue_env \
	-v $(shell pwd)/smartnet_tools:/smartnet_tools \
	-v $(shell pwd)/smartnet_rl:/smartnet_rl \
	-v $(SN_RESULTS_DIR):/results \
	--shm-size=10.06gb \
	queue_env

rllib-dev-run:
	sudo docker run -it --gpus all \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	-v $(shell pwd)/queue_env:/queue_env \
	-v $(shell pwd)/smartnet_tools:/smartnet_tools \
	-v $(shell pwd)/smartnet_rl:/smartnet_rl \
	-v $(shell pwd)/../../libs/ray:/ray \
	-v $(SN_RESULTS_DIR):/results \
	--shm-size=10.06gb \
	queue_env

prod-build:
	sudo docker build \
	-f docker/QueueEnvDockerfile \
	--target prod \
	--tag queue_env_prod \
	.

prod-run:
	sudo docker run -it --gpus all \
	-e WANDB_API_KEY=${WANDB_API_KEY} \
	--shm-size=10.06gb \
	queue_env_prod