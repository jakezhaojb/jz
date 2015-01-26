local SpatialMaxPoolingPos, parent = torch.class('nn.SpatialMaxPoolingPos', 'nn.Module')
--TODO temporary

function SpatialMaxPoolingPos:__init(kW, kH)
   parent.__init(self)
   self.kW = kW
   self.kH = kH
end

function SpatialMaxPoolingPos:updateOutput(input)
   if input:type() == 'torch.CudaTensor' then
      self.output = torch.CudaTensor()
      self.output_dx = torch.CudaTensor()
      self.output_dy = torch.CudaTensor()
      jz.SpatialMaxPoolingPos_updateOutput(self, input)
   else
      local inputSize = input:size()
      if inputSize:size() ~= 4 then
         print('Expected a 4D Tensor in SpatialMaxPooling')
         return nil
      end
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlanes = 3*inputSize[2] -- TODO
      local nOutputCols = math.floor(inputSize[4]/kW)
      local nOutputRows = math.floor(inputSize[3]/kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH

      self.output:resize(inputSize[1], nOutputPlanes, nOutputRows, nOutputCols):typeAs(input)

      for batch = 1,nBatches do
         for inplane = 1, inputSize[2] do
            local oi = 1
              for i = 1, maxH, kH do
                 local oj = 1
                 for j = 1, maxW, kW do
                    -- get the max one
                    local poolMax = 0
                    local dx = 0
                    local dy = 0
                    -- this is super-slow
                    for h = 1, kH do
                       for w = 1, kW do
                          if input[batch][inplane][i+h-1][j+w-1] > poolMax then
                             poolMax = input[batch][inplane][i+h-1][j+w-1]
                             dx = h
                             dy = w
                          end
                       end
                    end
                    self.output[batch][inplane*3-2][oi][oj] = poolMax
                    self.output[batch][inplane*3-1][oi][oj] = dx
                    self.output[batch][inplane*3][oi][oj] = dy
                    oj = oj + 1
                 end
                 oi = oi + 1
              end
         end
      end
   end
   return self.output
end


function SpatialMaxPoolingPos:updateGradInput(input, gradOutput)
   if input:type() == 'torch.CudaTensor' then
      jz.SpatialMaxPoolingPos(self, input, gradOutput)
   else
      self.gradInput = torch.Tensor():resizeAs(input):fill(0):typeAs(input)
      local inputSize = input:size()
      if inputSize:size() ~= 4 then
         print('Expected a 4D Tensor in SpatialMaxPooling')
         return nil
      end
      local kW = self.kW
      local kH = self.kH
      local nBatches = inputSize[1]
      local nOutputPlanes = 3*inputSize[2] -- TODO
      local nOutputCols = math.floor(inputSize[4]/kW)
      local nOutputRows = math.floor(inputSize[3]/kH)
      local maxW = nOutputCols * kW
      local maxH = nOutputRows * kH
      
      for batch = 1, nBatches do
         for inplane = 1, inputSize[2] do
            local dxPlane = self.output[batch][3*inplane-1]
            local dyPlane = self.output[batch][3*inplane]
            local gradOutputElem = gradOutput[batch][3*inplane-2]
            local oi = 1
            for i = 1, maxH, kH do
               local oj = 1
               for j = 1, maxW, kW do
                  self.gradInput[batch][inplane][i+dxPlane[oi][oj]-1][j+dyPlane[oi][oj]-1] = gradOutputElem[oi][oj]
                  oj = oj + 1
               end
               oi = oi + 1
            end
         end
      end
   end
   return self.gradInput
end
