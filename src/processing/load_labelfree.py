import os

from cellpose import models, core, io, plot

'''
Cell segmentation performed using Cellpose-SAM [1]
References:
[1] Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). 
    Cellpose: a generalist algorithm for cellular segmentation. 
    Nature Methods, 18(1), 100–106. https://doi.org/10.1038/s41592-020-01018-x
'''
def load_cellpose_model(models_dir=None):
    # io.logger_setup()  # run this to get printing of progress
    if not core.use_gpu():
        raise RuntimeError("GPU not available. Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA installed.")

    if models_dir is not None:
        pretrained_path = os.path.join(models_dir, 'cellpose', 'mcwell_bf')
        model = models.CellposeModel(gpu=True, pretrained_model=pretrained_path)
        print(f"Loaded retrained model {pretrained_path}.")
    else:
        model = models.CellposeModel(gpu=True)  # Use 'sam' for Cellpose-SAM
    return model
