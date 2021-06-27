## Grasp Verification Datasets

### dataset_openclose_678
Classes: **grasp, notgrasped** <br />
image size: 128x128 <br />
augmented: no <br />

Images are taken with JeVois smart camera mounted on the YouBot gripper.
Contains images of grasped and not grasped situations of different objects and with different backgrounds and lightings. <br />
Contains 2134 items for grasped, 2191 items for notgrasped. Training this dataset will result in good accuracy. (Tested with fine-tuned MobileNets, single block ResNet)

### dataset_small
Classes: **grasp, notgrasped** <br />
image size: 128x128 <br />
augmented: no <br />

Images are taken with JeVois smart camera mounted on the YouBot gripper.
Contains images of grasped and not grasped situations of different objects and with different backgrounds and lightings. <br />
Contains 300 items for grasped, 300 items for notgrasped. Training this dataset will result in bad accuracy. (Tested with fine-tuned MobileNets, single block ResNet)