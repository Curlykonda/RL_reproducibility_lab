from models.DQN import DQN
from models.DQN_HER import DQN_HER
from models.DQN_PER import DQN_PER


def main():
    # DQN.run("MountainCar-v0")
    DQN_HER.run("MountainCar-v0")
    # DQN_PER.run("MountainCar-v0")


if __name__ == "__main__":
    main()
