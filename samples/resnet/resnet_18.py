import torch
import torchvision

if __name__ == "__main__":
  # load model
  # details: https://pytorch.org/vision/stable/models.html

  print( 'loading model' )
  l_model = torchvision.models.resnet18( weights = "ResNet18_Weights.DEFAULT" )

  print( l_model )

  # freeze model
  l_model = l_model.eval()
  for l_pa in l_model.parameters():
    l_pa.requires_grad = False

  # print model size after first ops
  # see: https://github.com/pytorch/vision/blob/4af683107f967d4d435be4180020989dd8c3019c/torchvision/models/resnet.py#L266
  l_input = torch.randn( [1, 3, 224, 224] )
  l_tmp = l_model.conv1( l_input )
  l_tmp = l_model.bn1( l_tmp )
  l_tmp = l_model.relu( l_tmp )
  l_tmp = l_model.maxpool( l_tmp )
  print( 'size of activations after first ops:' )
  print( l_tmp.size() )

  # export first layer
  l_model_ts = torch.jit.script( l_model.layer1 )
  print( 'saving layer1' )
  l_model_ts.save( "resnet18_layer1_fp32.pt" )

  # optimize first layer
  l_model_ts = torch.jit.optimize_for_inference( l_model_ts )
  print( l_model_ts.graph )

  print( 'benchmarking first layer' )
  l_model_ts( l_tmp )

  with torch.profiler.profile( activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True ) as l_prof:
    with torch.profiler.record_function( "inference" ):
      for l_iter in range(64):
        l_model_ts( l_tmp )

  print( l_prof.key_averages().table( sort_by   = "cpu_time_total",
                                      row_limit = 10 ) )

  # export second layer
  print( 'saving layer2' )
  l_model_ts = torch.jit.script( l_model.layer2 )
  l_model_ts.save( "resnet18_layer2_fp32.pt" )

  print( 'benchmarking second layer' )
  l_model_ts( l_tmp )

  with torch.profiler.profile( activities = [torch.profiler.ProfilerActivity.CPU], record_shapes = True ) as l_prof:
    with torch.profiler.record_function( "inference" ):
      for l_iter in range(64):
        l_model_ts( l_tmp )

  print( l_prof.key_averages().table( sort_by   = "cpu_time_total",
                                      row_limit = 10 ) )