local SpatialMaxPoolingPos, parent = torch.class('jz.SpatialMaxPoolingPos', 'nn.Module')

function SpatialMaxPoolingPos:__init(kW, kH)
   parent.__init(self)
   self.kW = kW
   self.kH = kH
end

function SpatialMaxPoolingPos:updateOutput(input)
  -- TODO see how torch7 tackles this
   if input:dim() ~= 4 then
      if input:dim() == 3 then
         input = input:float():reshape(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
      else
         print('Expected a 3D/4D Tensor in SpatialMaxPooling')
         return nil
      end
   end
   local inputSize = input:size()
   if input:type() == 'torch.CudaTensor' then
      self.output_p = torch.CudaTensor()
      self.output_dx = torch.CudaTensor()
      self.output_dy = torch.CudaTensor()
      jz.SpatialMaxPoolingPos_updateOutput(self, input)
      join_table = nn.JoinTable(2)
      join_table:cuda()
      self.output = join_table:forward({self.output_p, self.output_dx, self.output_dy})
   else
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlanes = inputSize[2]
      local nOutputCols = math.floor(inputSize[4]/kW)
      local nOutputRows = math.floor(inputSize[3]/kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH

      self.output = self.output:resize(inputSize[1], 3*nOutputPlanes, nOutputRows, nOutputCols):typeAs(input)
      self.output_p = torch.Tensor(inputSize[1], nOutputPlanes, nOutputRows, nOutputCols):typeAs(input)
      self.output_dx = torch.Tensor(inputSize[1], nOutputPlanes, nOutputRows, nOutputCols):typeAs(input)
      self.output_dy = torch.Tensor(inputSize[1], nOutputPlanes, nOutputRows, nOutputCols):typeAs(input)

      for batch = 1,nBatches do
         for inplane = 1, inputSize[2] do
            local oi = 1
              for i = 1, maxH, kH do
                 local oj = 1
                 for j = 1, maxW, kW do
                    -- get the max one
                    local poolMax = 0
                    local dy = 0
                    local dx = 0
                    -- this is super-slow
                    for h = 1, kH do
                       for w = 1, kW do
                          if input[batch][inplane][i+h-1][j+w-1] > poolMax then
                             poolMax = input[batch][inplane][i+h-1][j+w-1]
                             dy = h
                             dx = w
                          end
                       end
                    end
                    self.output_p[batch][inplane][oi][oj] = poolMax
                    self.output_dy[batch][inplane][oi][oj] = dy
                    self.output_dx[batch][inplane][oi][oj] = dx
                    oj = oj + 1
                 end
                 oi = oi + 1
              end
         end
      end
      join_table = nn.JoinTable(2)
      self.output = join_table:forward({self.output_p, self.output_dx, self.output_dy})
   end
   self.output_p = nil
   self.output_dx = nil
   self.output_dy = nil
   collectgarbage()
   return self.output
end


function SpatialMaxPoolingPos:updateGradInput(input, gradOutput)
   if input:dim() ~= 4 then
      if input:dim() == 3 then
         input = input:resize(1, input:size(1), input:size(2), input:size(3)):typeAs(input)
      else
         print('Expected a 3D/4D Tensor in SpatialMaxPooling')
         return nil
      end
   end
   inputSize = input:size()
   local nOutputPlanes = inputSize[2]
   self.output_p = self.output[{ {},{1, nOutputPlanes},{},{}  }]
   self.output_dx = self.output[{ {},{nOutputPlanes+1, 2*nOutputPlanes},{},{}  }]
   self.output_dy = self.output[{ {},{2*nOutputPlanes+1, 3*nOutputPlanes},{},{}  }]
   if input:type() == 'torch.CudaTensor' then
      gradOutput_p = gradOutput[{ {},{1,nOutputPlanes},{},{} }]
      jz.SpatialMaxPoolingPos_updateGradInput(self, input, gradOutput_p)
   else
      self.gradInput = torch.Tensor():resizeAs(input):fill(0):typeAs(input)
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputCols = math.floor(inputSize[4]/kW)
      local nOutputRows = math.floor(inputSize[3]/kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH
      local gradOutput_p = gradOutput[{ {},{1, nOutputPlanes},{},{}  }]
      local gradOutput_dx = gradOutput[{ {},{nOutputPlanes+1, 2*nOutputPlanes},{},{}  }]
      local gradOutput_dy = gradOutput[{ {},{2*nOutputPlanes+1, 3*nOutputPlanes},{},{}  }]
      
      for batch = 1, nBatches do
         for inplane = 1, inputSize[2] do
            local dwPlane = self.output_dx[batch][inplane]
            local dhPlane = self.output_dy[batch][inplane]
            local gradOutputElem = gradOutput_p[batch][inplane]
            local oi = 1
            for i = 1, maxH, kH do
               local oj = 1
               for j = 1, maxW, kW do
                  self.gradInput[batch][inplane][i+dhPlane[oi][oj]-1][j+dwPlane[oi][oj]-1] = gradOutputElem[oi][oj]
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
   end
   self.output_p = nil
   self.output_dx = nil
   self.output_dy = nil
   collectgarbage()
   return self.gradInput
end
