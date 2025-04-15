from simulationloader import SimulationDataReader
from genomeToImage import GenomeImage
import tensorflow as tf
from mylogger import Logger
import os 
import multiprocessing
from functools import partial
from tqdm import tqdm  # for progress bar

def loader_helper(simPath, causalSamples, columns, simIndex):
    try:
        simData = SimulationDataReader(simPath).run(simIndex, causalSamples)
    except Exception as e:
        Logger('Error:', os.environ['LOGGER']).error(f"Simulation {simIndex} failed with error: {e}")
        return ([], [], [])
    
    individuals = simData['input'].shape[1]
    rows = int(individuals / columns)
    genomeImage = GenomeImage(rows, columns)

    X_local = []
    y_local = []
    pop_local = []
    mds_vector = simData['population']
    for index, sample in enumerate(simData['input']):
        try:
            image = genomeImage.transform_to_image(sample)
        except Exception as e:
            Logger('Error:', os.environ['LOGGER']).error(f"Transformation failed at simulation {simIndex}, sample {index}: {e}")
            continue  # Skip this sample if transformation fails

        X_local.append(image)
        label = int(simData['labels'][index])
        y_local.append(label)
        pop_local.append(mds_vector)
    return (X_local, y_local, pop_local)



class Dataset:
    BASEPATH = './simulation/data/'

    def __init__(self, total_simulations: int, causalSamples: int, columns: int, simPath: str = None):
        self.total_simulations = total_simulations
        self.causalSamples = causalSamples  # causal samples, not individuals 
        self.columns = columns
        self.simPath = simPath if simPath else self.BASEPATH
        self.X = None
        self.y = None
        self.pop = None
        
    def load_data(self):
        cpus = multiprocessing.cpu_count()
        Logger(f'Message:', os.environ['LOGGER']).info(f"Using {cpus} CPU cores for data loading.")
        with multiprocessing.Pool(cpus) as pool:
            ss = partial(loader_helper, self.simPath, self.causalSamples, self.columns)
            results = list(tqdm(pool.imap_unordered(ss, range(self.total_simulations)), total=self.total_simulations))

        # Flatten results
        X_all, y_all, pop_all = [], [], []
        for x_part, y_part, pop_part in results:
            X_all.extend(x_part)
            y_all.extend(y_part)
            pop_all.extend(pop_part)

        Logger(f'Message:', os.environ['LOGGER']).info("Converting lists to tensors...")
        self.X = tf.convert_to_tensor(X_all, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y_all, dtype=tf.int32)
        self.pop = tf.convert_to_tensor(pop_all, dtype=tf.float32)
        Logger(f'Message:', os.environ['LOGGER']).info("Conversion complete.")


if __name__ == '__main__':
    dataset = Dataset(total_simulations=10, causalSamples=4, columns=20)
    dataset.load_data()
    print("Data loaded successfully.")
    print("X shape:", dataset.X.shape)
    print("y shape:", dataset.y.shape)
