from simulationloader import SimulationDataReader
from genomeToImage import GenomeImage
import tensorflow as tf
from mylogger import Logger
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm

def loader_helper(simPath: str, causalSamples: int, columns: int, simIndex: int):
    """
    Load one simulation, convert each genome sample into a 2D image,
    collect its labels and population vector.
    """
    try:
        simData = SimulationDataReader(simPath).run(simIndex, causalSamples)
    except Exception as e:
        Logger('Error:', os.environ['LOGGER']).error(
            f"[sim {simIndex}] failed to load: {e}"
        )
        return None


    individuals = simData['input'].shape[1]
    rows = individuals // columns
    genomeImage = GenomeImage(rows, columns)

    X_local, y_local, pop_local = [], [], []
    mds_vector = simData['population']

    for idx, sample in enumerate(simData['input']):
        try:
            img = genomeImage.transform_to_image(sample)
        except Exception as e:
            Logger('Error:', os.environ['LOGGER']).error(
                f"[sim {simIndex}, sample {idx}] transform error: {e}"
            )
            continue

        X_local.append(img)
        y_local.append(int(simData['labels'][idx]))
        pop_local.append(mds_vector)

    return X_local, y_local, pop_local


class Dataset:
    BASEPATH = './simulation/data/'

    def __init__(
        self,
        total_simulations: int,
        causalSamples: int,
        columns: int,
        simPath: str = None,
        timeout: float = 30.0,
    ):
        self.total_simulations = total_simulations
        self.causalSamples = causalSamples
        self.columns = columns
        self.simPath = simPath or self.BASEPATH
        self.timeout = timeout

        # final tensors
        self.X: tf.Tensor = None
        self.y: tf.Tensor = None
        self.pop: tf.Tensor = None

    def load_data(self):
        """
        Parallel-load all simulations using a process pool, flatten results,
        and convert to TensorFlow tensors.
        """
        cpus = os.cpu_count() or 4
        Logger('Message:', os.environ['LOGGER']).info(
            f"Spawning {cpus} worker processes for data loading."
        )

        X_all, y_all, pop_all = [], [], []

        with ProcessPoolExecutor(max_workers=cpus) as executor:
            # submit all sims
            futures = {
                executor.submit(
                    loader_helper,
                    self.simPath,
                    self.causalSamples,
                    self.columns,
                    idx
                ): idx
                for idx in range(self.total_simulations)
            }

            for future in tqdm(
                as_completed(futures),
                total=self.total_simulations,
                desc="Loading simulations",
            ):
                sim_idx = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                except TimeoutError:
                    Logger('Error:', os.environ['LOGGER']).error(
                        f"[sim {sim_idx}] timed out after {self.timeout}s"
                    )
                    continue
                except Exception as e:
                    Logger('Error:', os.environ['LOGGER']).error(
                        f"[sim {sim_idx}] failed: {e}"
                    )
                    continue

                if result:
                    x_part, y_part, pop_part = result
                    X_all.extend(x_part)
                    y_all.extend(y_part)
                    pop_all.extend(pop_part)

        Logger('Message:', os.environ['LOGGER']).info(
            "All simulations loaded; converting lists to tensors."
        )
        self.X = tf.convert_to_tensor(X_all, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y_all, dtype=tf.int32)
        self.pop = tf.convert_to_tensor(pop_all, dtype=tf.float32)
        Logger('Message:', os.environ['LOGGER']).info("Tensors ready.")

    def as_tf_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """
        Wrap the in-memory tensors into a tf.data.Dataset
        with optional shuffling and batching.
        """
        ds = tf.data.Dataset.from_tensor_slices((self.X, self.y, self.pop))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.X))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


if __name__ == '__main__':
    # Example usage
    dataset = Dataset(
        total_simulations=100,
        causalSamples=4,
        columns=20,
        simPath=None,
        timeout=20.0
    )

    # 1) Load into memory and build tensors
    dataset.load_data()
    print("Loaded tensors:")
    print("  X:", dataset.X.shape)
    print("  y:", dataset.y.shape)
    print("  pop:", dataset.pop.shape)

    # 2) Create a tf.data pipeline for training
    tfds = dataset.as_tf_dataset(batch_size=64)
    for batch in tfds.take(1):
        x_batch, y_batch, pop_batch = batch
        print("Sample batch shapes:", x_batch.shape, y_batch.shape, pop_batch.shape)
