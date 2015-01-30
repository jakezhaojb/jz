require 'jz'

unit_test_SpatialMaxPoolingPos = function()
   print("--- Input --- \n")
   x = torch.rand(1,2,4,4)
   print(x)
   a = nn.Sequential()
   a:add(jz.SpatialMaxPoolingPos(2,2))
   y = a:forward(x)
   print("--- Output --- \n")
   print(y)
   --randGrad = torch.Tensor():resizeAs(y):typeAs(y):fill(1)
   randGrad = torch.rand(y:size()):typeAs(y)
   dfdx = a:backward(x, randGrad)
   print("--- Gradients --- \n")
   print(dfdx)
   print("--- CUDA --- \n")
   xc = x:cuda()
   b = a:clone()
   b:cuda()
   print("--- Output --- \n")
   yc = b:forward(xc)
   print(yc)
   --print("--- Output dx --- \n")
   --print(b:get(1).output_dx)
   --print("--- Output dy --- \n")
   --print(b:get(1).output_dy)
   print("--- Gradients --- \n")
   dfdxc = b:backward(xc, randGrad:cuda())
   print(dfdxc)
end


unit_test_SpatialMaxUnpoolingPos = function()
   print("--- Input --- ")
   x = torch.rand(1,2,4,4)
   --x = torch.rand(1,4,4) -- Different number of planes of Tensor
   print(x)
   a = nn.Sequential()
   a:add(jz.SpatialMaxPoolingPos(2,2))
   a:add(jz.SpatialMaxUnpoolingPos(2,2))
   print("--- Output after unPooling ---")
   y = a:forward(x)
   print(y)
   --randGrad = torch.Tensor():resizeAs(y):typeAs(y):fill(1)
   randGrad = torch.rand(y:size()):typeAs(y)
   print("--- Gradients --- ")
   dfdx = a:backward(x,randGrad)
   print(dfdx)
   print("--- CUDA --- \n")
   b = a:clone()
   b:cuda()
   xc = x:cuda()
   print("--- Output after Unpooling --- ")
   yc = b:forward(xc)
   print(yc)
   print("--- Gradients --- ")
   dfdxc = b:backward(xc, randGrad:cuda())
   print(dfdxc)
   print(" \n Gradient benchmark \n")
   print(randGrad)
end


--unit_test_SpatialMaxPoolingPos()
unit_test_SpatialMaxUnpoolingPos()
