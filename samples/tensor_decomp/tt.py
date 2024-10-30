##
# Uses a tensor train decomposition to compress the COIL-100 dataset.
# The ranks are set automatically based on the relative error (0.2).
#
# Additional information: https://tntorch.readthedocs.io/en/latest/tutorials/decompositions.html#Error-bounded-Decompositions
##
import argparse
import torch
import tntorch
import numpy
import opt_einsum

def download_coil100():
    import tensorflow_datasets
    import pandas

    coil100 = tensorflow_datasets.load('coil100')['train']
    coil100 = tensorflow_datasets.as_dataframe( coil100 )

    coil100 = coil100.sort_values( by=['object_id', 'angle'] )

    images = coil100['image']
    images = numpy.stack( images.values )

    # reshape the image tensor to (n_objects, n_angles, height, width, channels)
    images = images.reshape( (100, 72, 128, 128, 3) )

    numpy.save( "coil100.npy", images )

def to_image( tensor ):
    if not isinstance( tensor, numpy.ndarray ):
        tensor = tensor.numpy()
    tensor -= numpy.min( tensor )
    tensor /= numpy.max( tensor )
    tensor *= 255
    return tensor.astype( numpy.uint8 )

def print_metrics( tensor_decomp,
                   tensor_full ):
    print( "decomposition metrics:" )
    print( tensor_decomp )
    print( '  compression ratio: {}/{} = {:g}'.format( tensor_full.numel(), tensor_decomp.numcoef(), tensor_full.numel() / tensor_decomp.numcoef()) )
    print( '  relative error:', tntorch.relative_error(tensor_full, tensor_decomp) )
    print( '  RMSE:', tntorch.rmse(tensor_full, tensor_decomp) )
    print( '  R^2:', tntorch.r_squared(tensor_full, tensor_decomp) )

def decompose( data,
               verbose = True ):
    data = torch.tensor( data,
                         dtype = torch.float32 )

    if verbose:
        print( 'computing tensor decomposition' )
    tensor_decomp = tntorch.Tensor( data )
    tensor_decomp.round_tt( eps=0.2 )

    print_metrics( tensor_decomp,
                   data)

    if verbose:
        print( "shape of tucker core:",
               tensor_decomp.tucker_core().shape )

    if verbose:
        print( "shape and dtype of cores:" )
        for core in tensor_decomp.cores:
            print( '  ', core.shape )
            print( '  ', core.dtype )

        print( "shape and dtype of factors (if any):" )
        for factor in tensor_decomp.Us:
            if factor is not None:
                print( factor.shape )
                print( type(factor) )
                print( factor.dtype )

    return tensor_decomp

# check for main function
if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--download",
                         action = "store_true",
                         help = "download COIL-100 dataset" )
    parser.add_argument( "--plot",
                         action = "store_true",
                         help = "plot the first 5 original, auto reconstructed images and manual reconstructed images" )
    args = parser.parse_args()

    if args.download:
        print( 'downloading COIL-100 dataset' )
        download_coil100()

    print( 'loading coil100.npy' )
    images = numpy.load( "coil100.npy" ).astype( numpy.float32 )
    print( "dataset shape:", images.shape )

    tensor_decomp = decompose( images )

    # perform reconstruction manually
    print( "performing reconstruction manually" )
    result = torch.einsum( "af,fbg,gch,hdi,ie->abcde",
                           tensor_decomp.cores[0].squeeze(),
                           tensor_decomp.cores[1],
                           tensor_decomp.cores[2],
                           tensor_decomp.cores[3],
                           tensor_decomp.cores[4].squeeze() )
    result = result.squeeze()
    
    print( 'shape of manually reconstructed tensor:', result.shape )

    print( 'relative error (manual vs. tntorch):', tntorch.relative_error( tensor_decomp,
                                                                           result ) )

    print( 'opt_einsum contraction path:' )
    print( opt_einsum.contract_path( "af,fbg,gch,hdi,ie->abcde",
                                     tensor_decomp.cores[0].squeeze(),
                                     tensor_decomp.cores[1],
                                     tensor_decomp.cores[2],
                                     tensor_decomp.cores[3],
                                     tensor_decomp.cores[4].squeeze(),
                                     optimize = 'optimal' ) )

    if args.plot:
        # plot the first 5 original, auto reconstructed images and manual reconstructed images
        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i in range(5):
            ax = fig.add_subplot(5, 3, 3*i+1)
            ax.set_axis_off()
            ax.imshow(to_image(images[i,0]))
            if i == 0:
                ax.set_title("original")

            ax = fig.add_subplot(5, 3, 3*i+2)
            ax.imshow(to_image(tensor_decomp[i,0]))
            if i == 0:
                ax.set_title("auto reconstruction")

            ax = fig.add_subplot(5, 3, 3*i+3)
            ax.imshow(to_image(result[i,0]))
            if i == 0:
                ax.set_title("manual reconstruction")

        plt.show()