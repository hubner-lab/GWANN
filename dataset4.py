from simulationloader4 import SimulationDataReader
from genomeToImage import GenomeImage
import tensorflow as tf
from mylogger import Logger
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import numpy as np 
def loader_helper(simPath: str, sampledSitesIncludeCausals: int, columns: int, simIndex: int, mds:bool, trained_individuals:int):
    """
    Load one simulation, convert each genome sample into a 2D image,
    collect its labels and population vector.
    """
    try:
        simData = SimulationDataReader(simPath, mds).run(simIndex, sampledSitesIncludeCausals)
    except Exception as e:
        Logger(f'Message:', f"{os.environ['LOGGER']}").error(
            f"[sim {simIndex}] failed to load: {e}"
        )
        return None


    individuals = simData['input'].shape[1]
    rows = trained_individuals // columns
    
    genomeImage = GenomeImage(rows, columns)

    X_local, y_local= [], []
    
    padding = False
    if trained_individuals > individuals:
        padding = True
    for idx, sample in enumerate(simData['input']):
        try:
            sample = np.array(sample)
            if padding:
                sample = np.pad(sample, (0, trained_individuals - individuals),  constant_values=-10)
            else:
                sample = sample[:individuals]
            img = genomeImage.transform_to_image(sample)
        except Exception as e:
            Logger('Message:', os.environ['LOGGER']).error(
                f"[sim {simIndex}, sample {idx}] transform error: {e}"
            )
            continue

        X_local.append(img)
        y_local.append(int(simData['labels'][idx]))

    return X_local, y_local


class Dataset:
    BASEPATH = './simulation/data/'

    def __init__(
        self,
        total_simulations: int,
        sampledSitesIncludeCausals: int,
        columns: int,
        mds:bool,
        individuals:str,
        simPath: str = None,
        timeout: float = 30.0,
    ):

        self.total_simulations = total_simulations
        self.sampledSitesIncludeCausals = sampledSitesIncludeCausals
        self.columns = columns
        self.simPath = simPath or self.BASEPATH
        self.timeout = timeout
        self.mds = mds
        self.individuals = individuals
        self.X: tf.Tensor = None
        self.y: tf.Tensor = None
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")

    def load_data(self):
        """
        Parallel-load all simulations using a process pool, flatten results,
        and convert to TensorFlow tensors.
        """
        cpus = os.cpu_count() or 4
        self.logger.info(
            f"Spawning {cpus} worker processes for data loading."
        )

        X_all, y_all = [], []
        # loader_helper(self.simPath, self.sampledSitesIncludeCausals, self.columns, 0) # for debug
        self.logger.debug(f"Running simulation with mds={self.mds}")
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            # submit all sims
            futures = {
                executor.submit(
                    loader_helper,
                    self.simPath,
                    self.sampledSitesIncludeCausals,
                    self.columns,
                    idx,
                    self.mds,
                    self.individuals
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
                    self.logger.error(
                        f"[sim {sim_idx}] timed out after {self.timeout}s"
                    )
                    continue
                except Exception as e:
                    self.logger.error(
                        f"[sim {sim_idx}] failed: {e}"
                    )
                    continue

                if result:
                    x_part, y_part = result
                    # self.logger.info(f"[sim {sim_idx}] loaded {len(x_part)} samples.")
                    X_all.extend(x_part)
                    y_all.extend(y_part)

        Logger('Message:', os.environ['LOGGER']).info(
            "All simulations loaded; converting lists to tensors."
        )
        self.X = tf.convert_to_tensor(X_all, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y_all, dtype=tf.int32)
        self.logger.info("Tensors ready.")
        self.logger.debug(f"Total samples: {len(self.X)}")
        self.logger.debug(f"Total labels: {len(self.y)}")
        total_causal = len(self.y[self.y == 1])
        total_non_causal_mds_include = len(self.y[self.y == 0])
        self.logger.debug(f"Total true labels: {total_causal}")
        self.logger.debug(f"Total false labels: {total_non_causal_mds_include}")
        self.logger.debug(f"Sampling rate: { int(total_causal / total_causal)}:{int(total_non_causal_mds_include/total_causal)}")




if __name__ == '__main__':
    # Example usage
    dataset = Dataset(
        total_simulations=100,
        sampledSitesIncludeCausals=2,
        columns=20,
        individuals=300,
        mds=False,
        simPath="/mnt/data/amir/GWANN-TEST/GWANN/simulation/data",
        timeout=20.0,
    )

    # 1) Load into memory and build tensors
    dataset.load_data()
    print("Loaded tensors:")
    print("  X:", dataset.X.shape)
    print("  y:", dataset.y.shape)
    print("  pop:", dataset.pop.shape)

 