require 'jz'

unit_test_SpatialMaxPoolingPos = function()
   print("--- Input --- \n")
   x = torch.rand(3,2,8,8)
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
   print("-- Discrepency -- ")
   print(dfdxc:add(dfdx:cuda():mul(-1)):norm())
   print(yc:add(y:cuda():mul(-1)):norm())
end


unit_test_SpatialMaxUnpoolingPos = function()
   print("--- Input --- ")
   x = torch.rand(3,2,8,8)
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
   print("-- Discrepency -- ")
   print(dfdxc:add(dfdx:cuda():mul(-1)):norm())
   print(yc:add(y:cuda():mul(-1)):norm())
end

unit_test_SpatialMlpUnPooling = function()
   print("--- Input --- ")
   x = torch.rand(3,2,8,8)
   print(x)
   a = nn.Sequential()
   a:add(jz.SpatialMlpUnPooling(2,2,2,2))
   print("--- weight ---")
   print(a:get(1).weight)
   print("--- Output after unPooling ---")
   y = a:forward(x)
   print(y)
   randGrad = torch.rand(y:size()):typeAs(y)
   -- Update parameters
   _, grad = a:getParameters()
   a:zeroGradParameters()
   print("--- Gradients --- ")
   dfdx = a:backward(x,randGrad)
   print(dfdx)
   --print("-- Gradient benchmark ---")
   --local sum_weight = torch.Tensor(x:size(2)):fill(0)
   --for j = 1, x:size(2) do
   --   sum_weight[j] = a:get(1).weight[j]:sum()
   --end
   --print(sum_weight)
   print("--- update parameters --- ")
   print(grad)
   grad_cpu = grad:clone()
   dfdx_cpu = dfdx:clone()
   y_cpu = y:clone()
   print("------")
   print("--- CUDA ---\n")
   print("------")
   xx = x:cuda()
   a:cuda()
   print("--- weight ---")
   print(a:get(1).weight)
   print("--- Output after unPooling ---")
   y = a:forward(xx)
   print(y)
   randGrad = randGrad:cuda()
   -- Update parameters
   _, grad = a:getParameters()
   a:zeroGradParameters()
   print("--- Gradients --- ")
   dfdx = a:backward(xx,randGrad)
   print(dfdx)
   print("--- update parameters --- ")
   print(grad)
   print("-- Discrepency -- ")
   print(grad:add(grad_cpu:cuda():mul(-1)):norm())
   print(dfdx:add(dfdx_cpu:cuda():mul(-1)):norm())
   print(y:add(y_cpu:cuda():mul(-1)):norm())
end


unit_test_SpatialMlpPooling = function()
   print("--- Input --- ")
   x = torch.rand(3,2,8,8)
   print(x)
   a = nn.Sequential()
   a:add(jz.SpatialMlpPooling(2,2,2,2))
   print("--- weight ---")
   print(a:get(1).weight)
   print("--- output benchmark --- ")
   print(a:get(1).weight[1]:sum() .. ' ' .. a:get(1).weight[2]:sum())
   print("\n")
   print("--- Output after Pooling ---")
   y = a:forward(x)
   print(y)
   randGrad = torch.rand(y:size()):typeAs(y)
   -- Update parameters
   _, grad = a:getParameters()
   a:zeroGradParameters()
   print("--- Gradients --- ")
   dfdx = a:backward(x,randGrad)
   print(dfdx)
   print("--- update parameters --- ")
   print(grad)
   grad_cpu = grad:clone()
   dfdx_cpu = dfdx:clone()
   y_cpu = y:clone()
   print("------")
   print("--- CUDA ---\n")
   print("------")
   xx = x:cuda()
   a:cuda()
   print("--- weight ---")
   print(a:get(1).weight)
   print("--- Output after Pooling ---")
   y = a:forward(xx)
   print(y)
   randGrad = randGrad:cuda()
   -- Update parameters
   _, grad = a:getParameters()
   a:zeroGradParameters()
   print("--- Gradients --- ")
   dfdx = a:backward(xx,randGrad)
   print(dfdx)
   print("--- update parameters --- ")
   print(grad)
   print("-- Discrepency -- ")
   print(grad:add(grad_cpu:cuda():mul(-1)):norm())
   print(dfdx:add(dfdx_cpu:cuda():mul(-1)):norm())
   print(y:add(y_cpu:cuda():mul(-1)):norm())
end

--unit_test_SpatialMaxPoolingPos()
--unit_test_SpatialMaxUnpoolingPos()
unit_test_SpatialMlpUnPooling()
--unit_test_SpatialMlpPooling()
