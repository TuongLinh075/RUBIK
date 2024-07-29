import json
import os

# Disable TensorFlow warnings and set environment variables to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import copy
from tensorflow import keras
import random
from twophase import SolutionManager

stickerToFace = {
    0: "D",
    1: "U",
    2: "F",
    3: "B",
    4: "L",
    5: "R"
}

minScrambleLen = 1
maxScrambleLen = 25

# Define the file extension for saved datasets
fileExt = ".npy"

# Possible cube turns
turns = ["D", "D2", "D'", "U", "U2", "U'", "F", "F2", "F'",
         "B", "B2", "B'", "L", "L2", "L'", "R", "R2", "R'"]

# Hyperparameters
trainingSize = 50000000
batchSize = 512
epochs = 10
numFiles = 4
checkpointPath = "checkpoint.keras"

# Define the size of the cube representation and hidden layer size for the model
maxInputLen = 54
hiddenSize = 128


def solve(cube_string, max_length=25, max_time=10):
    """
    Solve the cube specified by cube_string, return the first solution found
    as long as max_time not exceeded.
    """
    sm = SolutionManager(cube_string)
    solution = sm.solve(max_length, time.time() + max_time)
    if isinstance(solution, str):
        return solution
    elif solution == -2:
        raise RuntimeError("max_time exceeded, no solution found")
    elif solution == -1:
        raise RuntimeError("no solution found, try increasing max_length")
    raise RuntimeError(
        f"SolutionManager.solve: unexpected return value {solution}"
    )


def _randomlyScrambleCube(cube):
    for _ in range(random.randint(minScrambleLen, maxScrambleLen)):
        index = random.randint(0, len(turns) - 1)
        cube(turns[index])

    # Make sure cube isn't solved
    while cube.isSolved():
        index = random.randint(0, len(turns) - 1)
        cube(turns[index])

    return cube


def _getDataFromSolution(cube, solution):
    moves = solution.split()
    res = np.zeros((len(moves), 55))

    for i in range(len(moves)):
        row = cube.stickers.flatten()
        row = np.append(row, [turns.index(moves[i])], axis=0)
        res[i] = row

        cube(moves[i])

    return res


def _toStickerString(stickers):
    # Shifts 6x3x3 tensor from default representation to twophase representation
    #   (note that twophasel.solve() requires different sticker order)
    # I.e.  1            0
    #     4 2 5 3  =>  4 2 1 5
    #       0            3
    def toTwoPhase(stickers):
        return stickers[[1, 5, 2, 0, 4, 3], :, :]

    # Converts index of sticker to face character
    def indexToFace(index):
        return stickerToFace[index]

    stickerList = toTwoPhase(stickers).flatten()
    stickerList = map(indexToFace, stickerList)
    return "".join(stickerList)


def _getSolution(cube):
    return solve(_toStickerString(cube.stickers))


def randomScrambles():
    cube = Cube()
    cube = _randomlyScrambleCube(cube)

    # Attempt to get solution
    try:
        solution = _getSolution(cube)
    # Get another scramble if solution cannot be found
    except RuntimeError:
        print("Solution not found. Attempting another scramble.")
        return randomScrambles()

    return _getDataFromSolution(cube, solution)


def getRandomScrambles(iterations):
    res = np.zeros((0, 55))  # 54 stickers + 1 solution move
    while res.shape[0] < iterations:
        print("Training examples generated: " +
              str(res.shape[0]) + "/" + str(iterations), end="\r")
        res = np.concatenate((res, randomScrambles()), axis=0)

    return res[:iterations]


# Generate random training data for the cube solver
def generate_data(m, num_files=1, file_path_base=""):
    for i in range(num_files):
        print("Dataset for file #" + str(i) + " is generating.")
        data = getRandomScrambles(int(m / numFiles))
        np.save(file_path_base + str(i) + fileExt, data)
        print("Dataset for file #" + str(i) + " is saved.")


# Define the Cube class to handle cube manipulations
class Cube:
    sideLen = 3
    turnMap = {}

    def __init__(self, stickers=np.zeros([6, 3, 3])):
        self.stickers = np.empty([6, 3, 3])

        # Initialize stickers with default values if not provided
        if np.array_equal(np.zeros([6, 3, 3]), stickers):
            for i in range(6):
                self.stickers[i, :, :] = np.full([self.sideLen, self.sideLen], i)
        else:
            self.stickers = copy.copy(stickers)

        # Map turns to their respective methods
        self.turnMap = {
            "D": self.rotate_d,
            "D2": self.rotate_d2,
            "D'": self.rotate_d_prime,
            "U": self.rotate_u,
            "U2": self.rotate_u2,
            "U'": self.rotate_u_prime,
            "F": self.rotate_f,
            "F2": self.rotate_f2,
            "F'": self.rotate_f_prime,
            "B": self.rotate_b,
            "B2": self.rotate_b2,
            "B'": self.rotate_b_prime,
            "L": self.rotate_l,
            "L2": self.rotate_l2,
            "L'": self.rotate_l_prime,
            "R": self.rotate_r,
            "R2": self.rotate_r2,
            "R'": self.rotate_r_prime,
        }

    def __call__(self, rotation):
        try:
            self.turnMap[rotation]()
        except KeyError:
            print("That turn doesn't exist!")

    def isSolved(self):
        for i in range(6):
            if not np.array_equal(self.stickers[i, :, :], np.full([self.sideLen, self.sideLen], i)):
                return False
        return True

    # ROTATIONS
    def rotate_d(self):
        self.stickers[0] = np.rot90(self.stickers[0], axes=(1, 0))
        self.stickers[[4, 3, 5, 2], 2] = self.stickers[[3, 5, 2, 4], 2]

    def rotate_d2(self):
        for _ in range(2):
            self.rotate_d()

    def rotate_d_prime(self):
        for _ in range(3):
            self.rotate_d()

    def rotate_u(self):
        self.stickers[1] = np.rot90(self.stickers[1], axes=(1, 0))
        self.stickers[[4, 2, 5, 3], 0] = self.stickers[[2, 5, 3, 4], 0]

    def rotate_u2(self):
        for _ in range(2):
            self.rotate_u()

    def rotate_u_prime(self):
        for _ in range(3):
            self.rotate_u()

    def rotate_f(self):
        self.stickers[2] = np.rot90(self.stickers[2], axes=(1, 0))
        tmp = copy.copy(self.stickers[4, :, 2])
        self.stickers[4, :, 2] = copy.copy(self.stickers[0, 0, :])
        self.stickers[0, 0, :] = copy.copy(np.flip(self.stickers[5, :, 0]))
        self.stickers[5, :, 0] = copy.copy(self.stickers[1, 2, :])
        self.stickers[1, 2, :] = np.flip(tmp)

    def rotate_f2(self):
        for _ in range(2):
            self.rotate_f()

    def rotate_f_prime(self):
        for _ in range(3):
            self.rotate_f()

    def rotate_b(self):
        self.stickers[3] = np.rot90(self.stickers[3], axes=(1, 0))
        tmp = copy.copy(self.stickers[5, :, 2])
        self.stickers[5, :, 2] = copy.copy(np.flip(self.stickers[0, 2, :]))
        self.stickers[0, 2, :] = copy.copy(self.stickers[4, :, 0])
        self.stickers[4, :, 0] = copy.copy(np.flip(self.stickers[1, 0, :]))
        self.stickers[1, 0, :] = tmp

    def rotate_b2(self):
        for _ in range(2):
            self.rotate_b()

    def rotate_b_prime(self):
        for _ in range(3):
            self.rotate_b()

    def rotate_l(self):
        self.stickers[4] = np.rot90(self.stickers[4], axes=(1, 0))
        tmp = copy.copy(self.stickers[3, :, 2])
        self.stickers[3, :, 2] = copy.copy(np.flip(self.stickers[0, :, 0]))
        self.stickers[0, :, 0] = copy.copy(self.stickers[2, :, 0])
        self.stickers[2, :, 0] = copy.copy(self.stickers[1, :, 0])
        self.stickers[1, :, 0] = np.flip(tmp)

    def rotate_l2(self):
        for _ in range(2):
            self.rotate_l()

    def rotate_l_prime(self):
        for _ in range(3):
            self.rotate_l()

    def rotate_r(self):
        self.stickers[5] = np.rot90(self.stickers[5], axes=(1, 0))
        tmp = copy.copy(self.stickers[2, :, 2])
        self.stickers[2, :, 2] = copy.copy(self.stickers[0, :, 2])
        self.stickers[0, :, 2] = copy.copy(np.flip(self.stickers[3, :, 0]))
        self.stickers[3, :, 0] = copy.copy(np.flip(self.stickers[1, :, 2]))
        self.stickers[1, :, 2] = tmp

    def rotate_r2(self):
        for _ in range(2):
            self.rotate_r()

    def rotate_r_prime(self):
        for _ in range(3):
            self.rotate_r()


# Loads data from specified input and output files, returns features and labels
def load_data(file_num=0, file_path_base=""):
    data = np.load(file_path_base + str(file_num) + fileExt).astype("float32")

    X_d = data[:, :54]
    Y_d = data[:, 54]

    return X_d, Y_d


# Defines model layers and compiles the model
def CubeModel():
    model = keras.Sequential([
        keras.layers.Input(shape=54),
        keras.layers.Dense(units=4096, activation="relu", name="dense0"),
        keras.layers.Dense(units=2048, activation="relu", name="dense1"),
        keras.layers.Dense(units=1024, activation="relu", name="dense2"),
        keras.layers.Dense(units=512, activation="relu", name="dense3"),
        keras.layers.Dense(units=256, activation="relu", name="dense4"),
        keras.layers.Dense(units=128, activation="relu", name="dense5"),
        keras.layers.Dense(units=64, activation="relu", name="dense6"),
        keras.layers.Dense(units=32, activation="relu", name="dense7"),
        keras.layers.Dense(units=18, activation="softmax", name="dense8"),
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    return model


# CubeSolver class that encapsulates model training and prediction methods
class CubeSolver:
    def __init__(self):
        self.model = CubeModel()

    def trainModel(self, loadPrev=True):
        if loadPrev:
            self.model.load_weights(checkpointPath)

        for i in range(numFiles):
            X, Y = load_data(i)
            callbacks = self.getCallbacks()
            self.model.fit(x=X, y=Y, epochs=epochs, batch_size=batchSize, validation_split=0.01, callbacks=callbacks)

        self.model.summary()

    def getCallbacks(self):
        checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", verbose=1,
                                                     save_weights_only=True, save_best_only=True)

        save_path = int(time.time())
        profile_path = f'logs/{save_path}/plugins/profile'
        if not os.path.exists(profile_path):
            os.makedirs(f'logs/{save_path}/plugins/profile')
        earlyStopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1)
        tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(save_path))

        return [checkpoint, earlyStopping, tensorboard]

    def predict(self, stickers):
        c = Cube(stickers)
        s = np.reshape(c.stickers, (1, 54))
        soln = ""
        count = 0

        while not c.isSolved():
            count += 1
            if count > 50:
                return "Solution could not be found."
            pred = self.predictMove(s)
            soln += " " + pred
            c(pred)

        return soln

    def predictMove(self, stickers):
        pred = np.argmax(self.model.predict(stickers), axis=-1)[0]
        return turns[pred]


if __name__ == "__main__":
    generate_data(trainingSize, num_files=numFiles)

    solver = CubeSolver()
    solver.trainModel(loadPrev=True)
