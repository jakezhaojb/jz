require 'nn'

unit_test_SpatialMaxPoolingPos = function()
   dofile("SpatialMaxPoolingPos.lua")
   print("--- Input --- \n")
   x = torch.rand(1,2,4,4)
   print(x)
   a = nn.Sequential()
   a:add(nn.SpatialMaxPoolingPos(2,2))
   y = a:forward(x)
   print("--- Output --- \n")
   print(y)
   randGrad = torch.Tensor():resizeAs(y):typeAs(y):fill(1)
   dfdx = a:backward(x, randGrad)
   print("--- Gradients --- \n")
   print(dfdx)
end

unit_test_SpatialMaxPoolingPos()
