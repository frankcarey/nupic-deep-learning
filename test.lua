require 'nn'
require 'image'
local ParamBank = require 'ParamBank'

local SpatialConvolution = nn.SpatialConvolution
local SpatialConvolutionMM = nn.SpatialConvolutionMM
local SpatialMaxPooling = nn.SpatialMaxPooling

local cuda = false;

if cuda then
   require 'cunn'
   require 'cudnn'
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialConvolutionMM = cudnn.SpatialConvolution
   SpatialMaxPooling = cudnn.SpatialMaxPooling
end

-- OverFeat input arguements
network  = 'small' or 'big'

-- system parameters
local threads = 4
local offset  = 0

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(threads)
print('==> #threads:', torch.getnumthreads())


local function nilling(module)
   module.gradBias   = nil
   if module.finput then module.finput = torch.Tensor():typeAs(module.finput) end
   module.gradWeight = nil
   module.output     = torch.Tensor():typeAs(module.output)
   module.fgradInput = nil
   module.gradInput  = nil
end

local function netLighter(network)
   nilling(network)
   if network.modules then
      for _,a in ipairs(network.modules) do
         netLighter(a)
      end
   end
end


net = nn.Sequential()
local m = net.modules
if network == 'small' then
   print('==> init a small overfeat network')
   net:add(SpatialConvolution(3, 96, 11, 11, 4, 4))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(96, 256, 5, 5, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolutionMM(1024, 3072, 6, 6, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialConvolutionMM(3072, 4096, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
   net:add(nn.View(1000))
   net:add(nn.SoftMax())
   net = net:float()
   print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_0")
   ParamBank:read(        0, {96,3,11,11},    m[offset+1].weight)
   ParamBank:read(    34848, {96},            m[offset+1].bias)
   ParamBank:read(    34944, {256,96,5,5},    m[offset+4].weight)
   ParamBank:read(   649344, {256},           m[offset+4].bias)
   ParamBank:read(   649600, {512,256,3,3},   m[offset+8].weight)
   ParamBank:read(  1829248, {512},           m[offset+8].bias)
   ParamBank:read(  1829760, {1024,512,3,3},  m[offset+11].weight)
   ParamBank:read(  6548352, {1024},          m[offset+11].bias)
   ParamBank:read(  6549376, {1024,1024,3,3}, m[offset+14].weight)
   ParamBank:read( 15986560, {1024},          m[offset+14].bias)
   ParamBank:read( 15987584, {3072,1024,6,6}, m[offset+17].weight)
   ParamBank:read(129233792, {3072},          m[offset+17].bias)
   ParamBank:read(129236864, {4096,3072,1,1}, m[offset+19].weight)
   ParamBank:read(141819776, {4096},          m[offset+19].bias)
   ParamBank:read(141823872, {1000,4096,1,1}, m[offset+21].weight)
   ParamBank:read(145919872, {1000},          m[offset+21].bias)

elseif network == 'big' then
   print('==> init a big overfeat network')
   net:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolutionMM(96, 256, 7, 7, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(256, 512, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(512, 512, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(512, 1024, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   net:add(SpatialConvolutionMM(1024, 1024, 3, 3, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialMaxPooling(3, 3, 3, 3))
   net:add(SpatialConvolutionMM(1024, 4096, 5, 5, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialConvolutionMM(4096, 4096, 1, 1, 1, 1))
   net:add(nn.Threshold(0, 1e-6))
   net:add(SpatialConvolutionMM(4096, 1000, 1, 1, 1, 1))
   net:add(nn.View(1000))
   net:add(nn.SoftMax())
   net = net:float()
   print(net)

   -- init file pointer
   print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_1")
   ParamBank:read(        0, {96,3,7,7},      m[offset+1].weight)
   ParamBank:read(    14112, {96},            m[offset+1].bias)
   ParamBank:read(    14208, {256,96,7,7},    m[offset+4].weight)
   ParamBank:read(  1218432, {256},           m[offset+4].bias)
   ParamBank:read(  1218688, {512,256,3,3},   m[offset+8].weight)
   ParamBank:read(  2398336, {512},           m[offset+8].bias)
   ParamBank:read(  2398848, {512,512,3,3},   m[offset+11].weight)
   ParamBank:read(  4758144, {512},           m[offset+11].bias)
   ParamBank:read(  4758656, {1024,512,3,3},  m[offset+14].weight)
   ParamBank:read(  9477248, {1024},          m[offset+14].bias)
   ParamBank:read(  9478272, {1024,1024,3,3}, m[offset+17].weight)
   ParamBank:read( 18915456, {1024},          m[offset+17].bias)
   ParamBank:read( 18916480, {4096,1024,5,5}, m[offset+20].weight)
   ParamBank:read(123774080, {4096},          m[offset+20].bias)
   ParamBank:read(123778176, {4096,4096,1,1}, m[offset+22].weight)
   ParamBank:read(140555392, {4096},          m[offset+22].bias)
   ParamBank:read(140559488, {1000,4096,1,1}, m[offset+24].weight)
   ParamBank:read(144655488, {1000},          m[offset+24].bias)

end
-- close file pointer
ParamBank:close()

-- PUT IMAGE STUFF BACK
--
ImgLoader = require 'ImgLoader'
img_list = {
  "lena",
  "bee.jpg"
}

for key,file in pairs(img_list) do
  img = ImgLoader:load(file);
end

Predictions = require 'predictions'
--itorch.image(img)
output = net:forward(img)
pred_a = Predictions:new{output = output}
pred_a:init()
pred_a:sort()
print(pred_a:getN(3))

