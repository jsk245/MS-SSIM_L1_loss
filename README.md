## INFO
This is a jax implementation of the mixed MS-SSIM and L1 loss function described here: https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf

The MS-SSIM loss is different than standard losses like L1 and L2 since it takes into account regions of pixels rather than only comparing one pixel at a time. Researchers found that the MS-SSIM loss preserved contrast in high-frequency regions while the L1 loss helped preserve colors, making the combination of them a strong performing loss function for image restoration models.

## Implementation Details
This notebook contains two implementations of the loss. The vectorized version is slightly faster, but the difference is negligible since both are compatible with jit, and the unvectorized is probably easier to read. This takes two images in jax arrays with values in the range [-1.0, 1.0] that are of the shape (Nx1xHxWxC) and calculates the loss. The default values of the constants used in the loss are the ones reported to work well in the paper above.
