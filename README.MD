# Shallow Depth of Field in Portraits Using Generative Models
> **Code Available**: https://github.com/ZainNasrullah/cyclegan-research-simulating-depth-of-field

The aim of this work was to explore how generative models can be improved at the task of simulating a shallow depth of field. To this end, the original CycleGAN implementation is extended with additional loss terms and strategies to guide training. Technical details can be found in the pdf or presentation attached.

An abstract is presented below:
> In response to the recent use of machine learning to create shallow depth of field images, this paper explores a generative approach to this task. While preliminary work in unpaired image translation has already explored this topic, prior methods are not able to reliably preserve the subject of an image and also have not been extended to pictures featuring people. For these reasons, this work introduces a novel portrait dataset containing images with and without a shallow depth of field. It further establishes a baseline, for visual comparison, using
both traditional smoothing techniques and the prior research in generative models. To solve the issue of subject preservation, two methods are proposed that take advantage of semantic segmentation: introducing a mask loss and performing a mask overlay. The former involves
computing the loss between masks of the subject in a real image and its corresponding generated output. This methods works well in terms of improving preservative performance without impacting the model’s ability to smooth the background of an image. The overlay method
places a mask of the real image’s subject on top of the generator output. Instead of preservation, the generator focuses solely on smoothing the background while the segmentation mask preserves the subject. This method improves cycle-consistency but also introduces artifacts
into the generated image. Interestingly, combining these seemingly contradictory approaches during training yields the best result in terms of subject preservation (the mask overlay is not used at test time) at the expense of generating a few artifacts. Additionally, as an unintended consequence, this combined model began generating peculiar results where backgrounds of images were stylized or contained imagined objects.

Since this research built upon pre-existing works, files that were heavily modified include:
- Loss calculations and training visualizations added to pytorch-CycleGAN-and-pix2pix\models\cycle_gan_model.py (adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Data loaders and training options are also modified for use with mask losses and segmentation masks.
- Segmentation mask generation from DeepLabv3 (adapted from the deeplab demo https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb) and manual smoothing visualizaitons in pytorch-CycleGAN-and-pix2pix\deeplab\inference.py
- Flickr Image Downloader is adapted from https://github.com/nagash91/python-flickr-image-downloader and modified for use in this project (inclusion of camera type)
