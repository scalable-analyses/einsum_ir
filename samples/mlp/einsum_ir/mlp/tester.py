import torch

## Tests the model
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied.
#  @param io_model model which is tested.
#  @return summed loss over all test samples, number of correctly predicted samples.
def test( i_loss_func,
          io_data_loader,
          io_model ):
  # switch model to evaluation mode
  io_model.eval()

  l_loss_total = 0
  l_n_correct = 0

  with torch.no_grad():
    for (l_x, l_y) in io_data_loader:
      # compute prediction and loss of the batch
      l_prediction = io_model( l_x )
      l_loss = i_loss_func( l_prediction, l_y )

      # sum total loss and derive number of correct predictions
      l_loss_total += l_loss.item()
      l_n_correct += (l_prediction.argmax(1) == l_y).type(torch.float).sum().item()

  return l_loss_total, l_n_correct
