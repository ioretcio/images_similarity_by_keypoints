import numpy as np
import matplotlib.cm as cm
import torch
import argparse
import time 
from src.matching import Matching
from src.utils import make_matching_plot, read_image

torch.set_grad_enabled(False)

class Matcher:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.nkeypoints = 500
        config = {
            'superpoint': {
                'nms_radius': 4,  # SuperPoint Non Maximum Suppression (NMS) radius
                'keypoint_threshold': 0.005,  # SuperPoint keypoint detector confidence threshold
                'max_keypoints': self.nkeypoints  # Maximum number of keypoints detected by Superpoint
            },
            'superglue': {
                'weights': 'outdoor',  # SuperGlue weights
                'sinkhorn_iterations': 20,  # Number of Sinkhorn iterations performed by SuperGlue
                'match_threshold': 0.2  # SuperGlue match threshold
            }
        }
        self.matching = Matching(config).eval().to(self.device)
        self.resizeWidth = 1024 
        self.resizeHeight = 840

    def calculateSimilarity(self, filename1:str, filename2:str):
        image0, inp0, scales0 = read_image(filename1, self.device, [ self.resizeWidth,self.resizeHeight], 0, False)
        image1, inp1, scales1 = read_image(filename2, self.device, [ self.resizeWidth,self.resizeHeight], 0, False)

        
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        
        return (sum(conf)/self.nkeypoints)*5

def main():
    parser = argparse.ArgumentParser(description="Calculate similarity between two images")
    parser.add_argument("im1", type=str, help="Filename of the first image")
    parser.add_argument("im2", type=str, help="Filename of the second image")
    args = parser.parse_args()

    matcher = Matcher()


    start = time.time()
    similarity = matcher.calculateSimilarity(args.im1, args.im2)
    print("Similarity between the images:", similarity)
    print(time.time() - start, " seconds")



if __name__ == "__main__":
    main()