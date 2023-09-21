import torch
import matplotlib.pyplot as plt

## Converts an Fashion MNIST numeric id to a string.
#  @param i_id numeric value of the label.
#  @return string corresponding to the id.
def toLabel( i_id ):
  l_labels = [ "T-Shirt",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle Boot" ]

  return l_labels[i_id]

## Applies the model to the data and plots the data.
#  @param i_off offset of the first image.
#  @param i_stride stride between the images.
#  @param io_data_loader data loader from which the data is retrieved.
#  @param io_model model which is used for the predictions.
#  @param i_path_to_pdf optional path to an output file, i.e., nothing is shown at runtime.
def plot( i_off,
          i_stride,
          io_data_loader,
          io_model,
          i_path_to_pdf = None ):
  # switch to evaluation mode
  io_model.eval()

  # offset of the current batch
  l_batch_off = 0
  # id of the next printed image
  l_id_print = i_off

  # create pdf if required
  if( i_path_to_pdf != None ):
    import matplotlib.backends.backend_pdf
    l_pdf_file = matplotlib.backends.backend_pdf.PdfPages( i_path_to_pdf )

  with torch.no_grad():
    for (l_x, l_y) in io_data_loader:
      l_n_images = len(l_x)

      # infer predictions on entire batch
      l_prediction = io_model( l_x )
      l_prediction = l_prediction.argmax(1)

      # iterate over images in batch
      for l_im in range( l_n_images ):
        l_id = l_batch_off + l_im

        # print only if id matches
        if( l_id == l_id_print ):
          # derive labels
          l_label_true = toLabel( int(l_y[l_im]) )
          l_label_pred = toLabel( int(l_prediction[l_im]) )

          # plot
          plt.imshow( l_x[l_im][0],
                      cmap = 'gray' )
          plt.title( l_label_true + ' (true) / ' + l_label_pred + ' (predicted)' )

          if( i_path_to_pdf != None ):
            l_pdf_file.savefig()
          else:
            plt.show()

          l_id_print = l_id_print + i_stride

      l_batch_off = l_batch_off + l_n_images

  # close pdf if required
  if( i_path_to_pdf != None ):
    l_pdf_file.close()
