# Automatic Mixed Precision Training

The default datatype when training a pytorch model is float32 , this can be costly so we have a workaround without having to sacrifice the precision of the training data which is AMP

What is Floating points operations ?
1. Allows training deep learning models using a combination of 32-bit (float32) and 16-bit (float16) data types.
2. Shorter training time: Mixed precision can provide a 1.5x to 5.5x speedup compared to training in full 32-bit precision on NVIDIA GPUs
3. Lower memory requirements: The reduced precision allows training larger models, larger batches, or larger inputs within the same memory budget

### benefits : 
- Shorter training time;
- Lower memory requirements, enabling larger batch sizes, larger models, or larger inputs

### Components :
1. `torch.autocast`
2. `torch.cuda.amp.GradScaler`

### [AutoCast]

Instances of `torch.autocast` enable autocasting for chosen regions. Autocasting automatically chooses the precision for GPU operations to improve performance while maintaining accuracy.


### [Gradient Scaling]

If the forward pass for a particular op has `float16` inputs, the backward pass for that op will produce `float16` gradients. Gradient values with small magnitudes may not be representable in `float16`. These values will flush to zero (“underflow”), so the update for the corresponding parameters will be lost.

To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are then scaled by the same factor. In other words, gradient values have a larger magnitude, so they don’t flush to zero.



Additional resources :
1. https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
2. https://pytorch.org/docs/stable/notes/amp_examples.html
3. https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
4. https://medium.com/@furkangozukara/what-is-the-difference-between-fp16-and-bf16-here-a-good-explanation-for-you-d75ac7ec30fa
5. https://lih-verma.medium.com/pytorchs-magic-with-automatic-mixed-precision-b3bef6f4b1fd