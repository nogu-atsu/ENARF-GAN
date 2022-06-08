import numpy as np


class THUmanPrior:
    def __init__(self):
        self.is_blank = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1])

        self.num_bone = 19

        self.prev_seq = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 11, 9, 10,
                         11, 12, 13, 16, 17, 18, 20, 21, 22, 23, 24, 25]

        self.num_joint = self.num_bone  # same as num_bone for this class
        self.num_not_blank_bone = int(np.sum(self.is_blank == 0))  # number of bone which is not blank

        self.valid_keypoints = [i for i in range(len(self.is_blank)) if i not in self.prev_seq or self.is_blank[i] == 0]
        self.num_valid_keypoints = len(self.valid_keypoints)
