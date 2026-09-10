"""FlexScaleSR - satellite multispectral super-resolution (SIH hackathon project).

Modules
-------
data        : TIFF loading, band reordering, reflectance scaling, Dataset + DataLoader
model       : the FlexScaleSR network
losses      : Charbonnier + SAM + Gradient loss
metrics     : PSNR, SSIM, MAE, RMSE, SAM, ERGAS
train_utils : config, seeding, device, train/validate loops, checkpoints
visualize   : RGB / NIR rendering and comparison panels
inference   : run a trained model over a folder of LR tiles and save GeoTIFFs
"""
__version__ = "1.0.0"
