from simulationloader import SimulationDataReader
from genomeToImage import GenomeImage
import tensorflow as tf
import sys 
from mylogger import Logger
import os 
class Dataset:
    BASEPATH = './simulation/data/'

    def __init__(self, total_simulations: int, causalSamples: int, columns: int, simPath: str = None):
        self.total_simulations = total_simulations
        self.causalSamples = causalSamples # causal samples not individuals 
        self.columns = columns
        self.X_list = []  # Use a Python list to collect tensors
        self.y_list = []
        self.simPath = simPath if simPath else self.BASEPATH
    def load_data(self):        
        for simIndex in range(self.total_simulations):
            simData = SimulationDataReader(self.simPath).run(simIndex, self.causalSamples)
            self.createImages(simData)
            percent = int(((simIndex + 1) / self.total_simulations) * 100)
            sys.stdout.write(f'\rLoading simulations... {percent}% complete')
            sys.stdout.flush()

        # Convert lists to tensors
        Logger(f'Message:', os.environ['LOGGER']).info("Converting lists to tensors...")
        self.X = tf.convert_to_tensor(self.X_list, dtype=tf.float32)  # Adjust dtype if needed
        self.y = tf.convert_to_tensor(self.y_list, dtype=tf.int32)
        Logger(f'Message:', os.environ['LOGGER']).info("Conversion complete.")

    def createImages(self, simData: dict):
        individuals = simData['input'].shape[1]
        rows = int(individuals / self.columns)
        genomeImage = GenomeImage(rows, self.columns)

        for index, sample in enumerate(simData['input']):
            image = genomeImage.transform_to_image(sample)
            self.X_list.append(image)  # Append image to the list
            label = int(simData['labels'][index])
            self.y_list.append(label)  # Append label

if __name__ == '__main__':
    dataset = Dataset(total_simulations=10, causalSamples=4, columns=20)
    dataset.load_data()
    print("Data loaded successfully.")
    print("X shape:", dataset.X.shape)
    print("y shape:", dataset.y.shape)
