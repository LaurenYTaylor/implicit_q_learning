import ray
import time
from train_jsrl import main as main_jsrl, make_save_dir_remote

ray.init(num_cpus=5)

seeds = list(range(20))
data_size = [1000, 10000, 100000, 1000000]
def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")

def train(seed, n_data, save_dir):
    return main_jsrl(seed, n_data, save_dir)
@ray.remote
def run_training(seed, n_data, save_dir):
    if seed in [0,1,2,3]:
        return
    elif seed == 4 and n_data in [1000,10000]:
        return
    return train(seed, n_data, save_dir)

start = time.time()
#save_dir = make_save_dir_remote(False, "antmaze-umaze-v0")
save_dir = "logs//antmaze-umaze-v0_8_jsrl_ft"
object_references = [
    run_training.remote(seeds[i], data_size[j], save_dir) for i in range(len(seeds)) for j in range(len(data_size))
]

all_data = []
while len(object_references) > 0:
    finished, object_references = ray.wait(
        object_references, timeout=7.0
    )
    data = ray.get(finished)
    print_runtime(data, start)
    all_data.extend(data)

print_runtime(all_data, start)

#data = ray.get(object_references)
#print_runtime(data, start)
