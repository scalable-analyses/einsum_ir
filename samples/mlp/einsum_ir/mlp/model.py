import torch.nn

#  Simple MultiLayer Perceptron (MLP) with fixed dimensions.
#
#  The MLP is assumes a 28^2 input-image and 10 output classes.
#  These are the dimensions of the Fashion MNIST dataset.
class Model( torch.nn.Module ):
  ## Initializes the class.
  #  @param self object pointer.
  def __init__( self ):
    super( Model, self ).__init__()
    ## flattens the input
    self.m_flatten = torch.nn.Flatten()
    ## layers of the MLP: 4x(linear + relu) + 1x linear
    self.m_layers = torch.nn.Sequential( torch.nn.Linear( 28*28, 512 ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear( 512, 512 ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear( 512, 512 ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear( 512, 512 ),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear( 512, 10 ) )

  ## Forward pass with the given input.
  #  @param self object pointer.
  #  @param i_input input for the forward pass.
  #  @return output of the MLP.
  def forward( self,
               i_input ):
    l_flatten = self.m_flatten( i_input )
    l_result = self.m_layers( l_flatten )
    return l_result