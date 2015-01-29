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
   --print("--- Output dx --- \n")
   --print(a:get(1).output_dx)
   --print("--- Output dy --- \n")
   --print(a:get(1).output_dy)
   randGrad = torch.Tensor():resizeAs(y):typeAs(y):fill(1)
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
   dfdxc = b:backward(xc, randGrad:cuda())
   print("--- Gradients --- \n")
   print(dfdxc)
end


unit_test_SpatialMaxUnpoolingPos = function()
   print("--- Input --- ")
   x = torch.rand(1,2,4,4)
   print(x)
   print("--- Pooling stage ---\n ")
   a = nn.Sequential()
   a:add(jz.SpatialMaxPoolingPos(2,2))
   print("--- Output after pooling ---")
   y = a:forward(x)
   print(y)
   a:add(jz.SpatialMaxUnpoolingPos(2,2))
   print("--- Output after unPooling ---")
   y1 = a:forward(x)
   print(y1)
   randGrad = torch.Tensor():resizeAs(y1):typeAs(y1):fill(1)
   dfdx = a:backward(x,randGrad)
   dfdxp = a:get(2):backward(y,randGrad)
   print("--- Gradients of Unpooling --- ")
   print(dfdxp)
   print("--- Gradients --- ")
   print(dfdx)
   
end


--unit_test_SpatialMaxPoolingPos()
unit_test_SpatialMaxUnpoolingPos()
