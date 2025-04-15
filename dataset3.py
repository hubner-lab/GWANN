from simulationloader import SimulationDataReader
from genomeToImage import GenomeImage
import tensorflow as tf
from mylogger import Logger
import os 
import multiprocessing
from functools import partial
from tqdm import tqdm  # for progress bar

def loader_helper(simPath, causalSamples, columns, simIndex):
    simData = SimulationDataReader(simPath).run(simIndex, causalSamples)
    individuals = simData['input'].shape[1]
    rows = int(individuals / columns)
    genomeImage = GenomeImage(rows, columns)

    X_local = []
    y_local = []

    for index, sample in enumerate(simData['input']):
        image = genomeImage.transform_to_image(sample)
        
        X_local.append(image)
        label = int(simData['labels'][index])
        y_local.append(label)

    return (X_local, y_local)


class Dataset:
    BASEPATH = './simulation/data/'

    def __init__(self, total_simulations: int, causalSamples: int, columns: int, simPath: str = None):
        self.total_simulations = total_simulations
        self.causalSamples = causalSamples  # causal samples, not individuals 
        self.columns = columns
        self.simPath = simPath if simPath else self.BASEPATH
        self.X = None
        self.y = None

    def load_data(self):
        cpus = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpus) as pool:
            ss = partial(loader_helper, self.simPath, self.causalSamples, self.columns)
            results = list(tqdm(pool.imap_unordered(ss, range(self.total_simulations)), total=self.total_simulations))

        # Flatten results
        X_all, y_all = [], []
        for x_part, y_part in results:
            X_all.extend(x_part)
            y_all.extend(y_part)

        Logger(f'Message:', os.environ['LOGGER']).info("Converting lists to tensors...")
        self.X = tf.convert_to_tensor(X_all, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y_all, dtype=tf.int32)
        Logger(f'Message:', os.environ['LOGGER']).info("Conversion complete.")


if __name__ == '__main__':
    dataset = Dataset(total_simulations=10, causalSamples=4, columns=20)
    dataset.load_data()
    print("Data loaded successfully.")
    print("X shape:", dataset.X.shape)
    print("y shape:", dataset.y.shape)
