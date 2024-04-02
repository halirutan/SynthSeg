# v1
- adjusted grey values for CBS segmented regions
- rest of the regions keep SynthSeg defaults
- CBS resolution (0.5 -> 512, 512, 512), crop images
- image outpu shape: [256, 256, 256, 3], n_labels: 33
- 157TB, 40000 files

# v2
- croped high resolution training data 500um, label maps provided by Patrick
- 2 channels: t1w and pd -> shape: [256, 256, 256, 2] , see SynthSeg/data/cbs/t1w_pdw_config
- ~40 000 files a 8 training pairs -> ~320 000 pairs, n_labels: 33
- validation data from https://datashare.mpcdf.mpg.de/f/348054516 , see SynthSeg/data/cbs/t1w_pdw_config/create_test_tfrecord.ipynb
